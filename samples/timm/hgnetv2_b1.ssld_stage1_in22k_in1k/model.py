import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_stem1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_stem1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_last_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_last_conv_modules_2_parameters_scale_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_last_conv_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_stem1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_stem1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem1_modules_lab_parameters_scale_ = (
            L_self_modules_stem_modules_stem1_modules_lab_parameters_scale_
        )
        l_self_modules_stem_modules_stem1_modules_lab_parameters_bias_ = (
            L_self_modules_stem_modules_stem1_modules_lab_parameters_bias_
        )
        l_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem2a_modules_lab_parameters_scale_ = (
            L_self_modules_stem_modules_stem2a_modules_lab_parameters_scale_
        )
        l_self_modules_stem_modules_stem2a_modules_lab_parameters_bias_ = (
            L_self_modules_stem_modules_stem2a_modules_lab_parameters_bias_
        )
        l_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem2b_modules_lab_parameters_scale_ = (
            L_self_modules_stem_modules_stem2b_modules_lab_parameters_scale_
        )
        l_self_modules_stem_modules_stem2b_modules_lab_parameters_bias_ = (
            L_self_modules_stem_modules_stem2b_modules_lab_parameters_bias_
        )
        l_self_modules_stem_modules_stem3_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem3_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem3_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem3_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem3_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem3_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem3_modules_lab_parameters_scale_ = (
            L_self_modules_stem_modules_stem3_modules_lab_parameters_scale_
        )
        l_self_modules_stem_modules_stem3_modules_lab_parameters_bias_ = (
            L_self_modules_stem_modules_stem3_modules_lab_parameters_bias_
        )
        l_self_modules_stem_modules_stem4_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem4_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem4_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem4_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem4_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem4_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem4_modules_lab_parameters_scale_ = (
            L_self_modules_stem_modules_stem4_modules_lab_parameters_scale_
        )
        l_self_modules_stem_modules_stem4_modules_lab_parameters_bias_ = (
            L_self_modules_stem_modules_stem4_modules_lab_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        l_self_modules_head_modules_last_conv_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_last_conv_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_last_conv_modules_2_parameters_scale_ = (
            L_self_modules_head_modules_last_conv_modules_2_parameters_scale_
        )
        l_self_modules_head_modules_last_conv_modules_2_parameters_bias_ = (
            L_self_modules_head_modules_last_conv_modules_2_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_stem1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_stem1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_stem1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_stem1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=False)
        x_1 = None
        mul = l_self_modules_stem_modules_stem1_modules_lab_parameters_scale_ * x_2
        l_self_modules_stem_modules_stem1_modules_lab_parameters_scale_ = x_2 = None
        x_3 = mul + l_self_modules_stem_modules_stem1_modules_lab_parameters_bias_
        mul = l_self_modules_stem_modules_stem1_modules_lab_parameters_bias_ = None
        x_4 = torch._C._nn.pad(x_3, (0, 1, 0, 1), "constant", None)
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_ = None
        x_6 = torch.nn.functional.batch_norm(
            x_5,
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_5 = (
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_ = None
        x_7 = torch.nn.functional.relu(x_6, inplace=False)
        x_6 = None
        mul_1 = l_self_modules_stem_modules_stem2a_modules_lab_parameters_scale_ * x_7
        l_self_modules_stem_modules_stem2a_modules_lab_parameters_scale_ = x_7 = None
        x_8 = mul_1 + l_self_modules_stem_modules_stem2a_modules_lab_parameters_bias_
        mul_1 = l_self_modules_stem_modules_stem2a_modules_lab_parameters_bias_ = None
        x2 = torch._C._nn.pad(x_8, (0, 1, 0, 1), "constant", None)
        x_8 = None
        x_9 = torch.conv2d(
            x2,
            l_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2 = l_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_ = None
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_ = None
        x_11 = torch.nn.functional.relu(x_10, inplace=False)
        x_10 = None
        mul_2 = l_self_modules_stem_modules_stem2b_modules_lab_parameters_scale_ * x_11
        l_self_modules_stem_modules_stem2b_modules_lab_parameters_scale_ = x_11 = None
        x_12 = mul_2 + l_self_modules_stem_modules_stem2b_modules_lab_parameters_bias_
        mul_2 = l_self_modules_stem_modules_stem2b_modules_lab_parameters_bias_ = None
        x1 = torch.nn.functional.max_pool2d(
            x_4, 2, 1, 0, 1, ceil_mode=True, return_indices=False
        )
        x_4 = None
        x_13 = torch.cat([x1, x_12], dim=1)
        x1 = x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stem_modules_stem3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_stem_modules_stem3_modules_conv_parameters_weight_ = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem3_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem3_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem3_modules_bn_parameters_bias_ = None
        x_16 = torch.nn.functional.relu(x_15, inplace=False)
        x_15 = None
        mul_3 = l_self_modules_stem_modules_stem3_modules_lab_parameters_scale_ * x_16
        l_self_modules_stem_modules_stem3_modules_lab_parameters_scale_ = x_16 = None
        x_17 = mul_3 + l_self_modules_stem_modules_stem3_modules_lab_parameters_bias_
        mul_3 = l_self_modules_stem_modules_stem3_modules_lab_parameters_bias_ = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_stem_modules_stem4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_stem_modules_stem4_modules_conv_parameters_weight_ = None
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem4_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem4_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem4_modules_bn_parameters_bias_ = None
        x_20 = torch.nn.functional.relu(x_19, inplace=False)
        x_19 = None
        mul_4 = l_self_modules_stem_modules_stem4_modules_lab_parameters_scale_ * x_20
        l_self_modules_stem_modules_stem4_modules_lab_parameters_scale_ = x_20 = None
        x_21 = mul_4 + l_self_modules_stem_modules_stem4_modules_lab_parameters_bias_
        mul_4 = l_self_modules_stem_modules_stem4_modules_lab_parameters_bias_ = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=False)
        x_23 = None
        mul_5 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_
            * x_24
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_ = (
            x_24
        ) = None
        x_25 = (
            mul_5
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_
        )
        mul_5 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_ = (None)
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=False)
        x_27 = None
        mul_6 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_
            * x_28
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_ = (
            x_28
        ) = None
        x_29 = (
            mul_6
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_
        )
        mul_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_ = (None)
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=False)
        x_31 = None
        mul_7 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_
            * x_32
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_ = (
            x_32
        ) = None
        x_33 = (
            mul_7
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_
        )
        mul_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_ = (None)
        x_34 = torch.cat([x_21, x_25, x_29, x_33], dim=1)
        x_21 = x_25 = x_29 = x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=False)
        x_36 = None
        mul_8 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
            * x_37
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = (
            x_37
        ) = None
        x_38 = (
            mul_8
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        )
        mul_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = (None)
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_41 = torch.nn.functional.relu(x_40, inplace=False)
        x_40 = None
        mul_9 = (
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
            * x_41
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = (
            x_41
        ) = None
        x_42 = (
            mul_9
            + l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        )
        mul_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = (None)
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            64,
        )
        x_42 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=False)
        x_46 = None
        mul_10 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_
            * x_47
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_scale_ = (
            x_47
        ) = None
        x_48 = (
            mul_10
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_
        )
        mul_10 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_lab_parameters_bias_ = (None)
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.relu(x_50, inplace=False)
        x_50 = None
        mul_11 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_
            * x_51
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_scale_ = (
            x_51
        ) = None
        x_52 = (
            mul_11
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_
        )
        mul_11 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_lab_parameters_bias_ = (None)
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=False)
        x_54 = None
        mul_12 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_
            * x_55
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_scale_ = (
            x_55
        ) = None
        x_56 = (
            mul_12
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_
        )
        mul_12 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_lab_parameters_bias_ = (None)
        x_57 = torch.cat([x_44, x_48, x_52, x_56], dim=1)
        x_44 = x_48 = x_52 = x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=False)
        x_59 = None
        mul_13 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
            * x_60
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = (
            x_60
        ) = None
        x_61 = (
            mul_13
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        )
        mul_13 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = (None)
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=False)
        x_63 = None
        mul_14 = (
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
            * x_64
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = (
            x_64
        ) = None
        x_65 = (
            mul_14
            + l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        )
        mul_14 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_65 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        x_69 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=False)
        x_71 = None
        mul_15 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_
            * x_72
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_ = (
            x_72
        ) = None
        x_73 = (
            mul_15
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_
        )
        mul_15 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        x_75 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=False)
        x_77 = None
        mul_16 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_
            * x_78
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_ = (
            x_78
        ) = None
        x_79 = (
            mul_16
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_
        )
        mul_16 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        x_81 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=False)
        x_83 = None
        mul_17 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_
            * x_84
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_ = (
            x_84
        ) = None
        x_85 = (
            mul_17
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_
        )
        mul_17 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_86 = torch.cat([x_67, x_73, x_79, x_85], dim=1)
        x_67 = x_73 = x_79 = x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_89 = torch.nn.functional.relu(x_88, inplace=False)
        x_88 = None
        mul_18 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
            * x_89
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = (
            x_89
        ) = None
        x_90 = (
            mul_18
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        )
        mul_18 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = (None)
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=False)
        x_92 = None
        mul_19 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
            * x_93
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = (
            x_93
        ) = None
        x_94 = (
            mul_19
            + l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        )
        mul_19 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = (None)
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        x_96 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=False)
        x_98 = None
        mul_20 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_
            * x_99
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_ = (
            x_99
        ) = None
        x_100 = (
            mul_20
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_
        )
        mul_20 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        x_102 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_105 = torch.nn.functional.relu(x_104, inplace=False)
        x_104 = None
        mul_21 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_
            * x_105
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_ = (
            x_105
        ) = None
        x_106 = (
            mul_21
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_
        )
        mul_21 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            96,
        )
        x_108 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_111 = torch.nn.functional.relu(x_110, inplace=False)
        x_110 = None
        mul_22 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_
            * x_111
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_ = (
            x_111
        ) = None
        x_112 = (
            mul_22
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_
        )
        mul_22 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_113 = torch.cat([x_94, x_100, x_106, x_112], dim=1)
        x_100 = x_106 = x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_116 = torch.nn.functional.relu(x_115, inplace=False)
        x_115 = None
        mul_23 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_scale_
            * x_116
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_scale_ = (
            x_116
        ) = None
        x_117 = (
            mul_23
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_bias_
        )
        mul_23 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_lab_parameters_bias_ = (None)
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_120 = torch.nn.functional.relu(x_119, inplace=False)
        x_119 = None
        mul_24 = (
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_scale_
            * x_120
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_scale_ = (
            x_120
        ) = None
        x_121 = (
            mul_24
            + l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_bias_
        )
        mul_24 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_lab_parameters_bias_ = (None)
        x_122 = x_121 + x_94
        x_121 = x_94 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            512,
        )
        x_122 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_126 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_129 = torch.nn.functional.relu(x_128, inplace=False)
        x_128 = None
        mul_25 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_
            * x_129
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_scale_ = (
            x_129
        ) = None
        x_130 = (
            mul_25
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_
        )
        mul_25 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_132 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=False)
        x_134 = None
        mul_26 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_
            * x_135
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_scale_ = (
            x_135
        ) = None
        x_136 = (
            mul_26
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_
        )
        mul_26 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_138 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=False)
        x_140 = None
        mul_27 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_
            * x_141
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_scale_ = (
            x_141
        ) = None
        x_142 = (
            mul_27
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_
        )
        mul_27 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_lab_parameters_bias_ = (None)
        x_143 = torch.cat([x_124, x_130, x_136, x_142], dim=1)
        x_124 = x_130 = x_136 = x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_146 = torch.nn.functional.relu(x_145, inplace=False)
        x_145 = None
        mul_28 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_
            * x_146
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_scale_ = (
            x_146
        ) = None
        x_147 = (
            mul_28
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_
        )
        mul_28 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_lab_parameters_bias_ = (None)
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=False)
        x_149 = None
        mul_29 = (
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_
            * x_150
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_scale_ = (
            x_150
        ) = None
        x_151 = (
            mul_29
            + l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_
        )
        mul_29 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_lab_parameters_bias_ = (None)
        x_152 = torch.nn.functional.adaptive_avg_pool2d(x_151, 1)
        x_151 = None
        input_1 = torch.conv2d(
            x_152,
            l_self_modules_head_modules_last_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = (
            l_self_modules_head_modules_last_conv_modules_0_parameters_weight_
        ) = None
        input_2 = torch.nn.functional.relu(input_1, inplace=False)
        input_1 = None
        mul_30 = (
            l_self_modules_head_modules_last_conv_modules_2_parameters_scale_ * input_2
        )
        l_self_modules_head_modules_last_conv_modules_2_parameters_scale_ = (
            input_2
        ) = None
        input_3 = (
            mul_30 + l_self_modules_head_modules_last_conv_modules_2_parameters_bias_
        )
        mul_30 = l_self_modules_head_modules_last_conv_modules_2_parameters_bias_ = None
        x_153 = torch.nn.functional.dropout(input_3, 0.0, False, False)
        input_3 = None
        x_154 = x_153.flatten(1, -1)
        x_153 = None
        x_155 = torch._C._nn.linear(
            x_154,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_154 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_155,)
