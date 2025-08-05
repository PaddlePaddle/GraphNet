import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_last_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_stem_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem_modules_0_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_stem_modules_2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_stem_modules_2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_
        l_self_modules_head_modules_last_conv_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_last_conv_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_stem_modules_0_modules_conv_parameters_weight_
        ) = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_stem_modules_0_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.relu(x_1, inplace=False)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = (
            l_self_modules_stem_modules_stem_modules_1_modules_conv_parameters_weight_
        ) = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_stem_modules_1_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=False)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stem_modules_stem_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_stem_modules_stem_modules_2_modules_conv_parameters_weight_
        ) = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem_modules_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_stem_modules_2_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.relu(x_7, inplace=False)
        x_7 = None
        x_9 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
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
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=False)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
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
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=False)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
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
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=False)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.relu(x_20, inplace=False)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=False)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_27 = torch.nn.functional.relu(x_26, inplace=False)
        x_26 = None
        x_28 = torch.cat([x_9, x_12, x_15, x_18, x_21, x_24, x_27], dim=1)
        x_9 = x_12 = x_15 = x_18 = x_21 = x_24 = x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=False)
        x_30 = None
        x_32 = x_31.mean((2, 3), keepdim=True)
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = (None)
        x_34 = torch.sigmoid(x_33)
        x_33 = None
        input_1 = torch.mul(x_31, x_34)
        x_31 = x_34 = None
        x_35 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        input_1 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_37 = torch.conv2d(
            x_36,
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
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=False)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
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
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=False)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
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
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=False)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_48 = torch.nn.functional.relu(x_47, inplace=False)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.relu(x_50, inplace=False)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=False)
        x_53 = None
        x_55 = torch.cat([x_36, x_39, x_42, x_45, x_48, x_51, x_54], dim=1)
        x_36 = x_39 = x_42 = x_45 = x_48 = x_51 = x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=False)
        x_57 = None
        x_59 = x_58.mean((2, 3), keepdim=True)
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = (None)
        x_61 = torch.sigmoid(x_60)
        x_60 = None
        input_2 = torch.mul(x_58, x_61)
        x_58 = x_61 = None
        x_62 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            512,
        )
        input_2 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_66 = torch.nn.functional.relu(x_65, inplace=False)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_69 = torch.nn.functional.relu(x_68, inplace=False)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=False)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=False)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=False)
        x_80 = None
        x_82 = torch.cat([x_63, x_66, x_69, x_72, x_75, x_78, x_81], dim=1)
        x_63 = x_66 = x_69 = x_72 = x_75 = x_78 = x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=False)
        x_84 = None
        x_86 = x_85.mean((2, 3), keepdim=True)
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = (None)
        x_88 = torch.sigmoid(x_87)
        x_87 = None
        input_3 = torch.mul(x_85, x_88)
        x_85 = x_88 = None
        x_89 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=False)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=False)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=False)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=False)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=False)
        x_105 = None
        x_107 = torch.cat([input_3, x_91, x_94, x_97, x_100, x_103, x_106], dim=1)
        x_91 = x_94 = x_97 = x_100 = x_103 = x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_110 = torch.nn.functional.relu(x_109, inplace=False)
        x_109 = None
        x_111 = x_110.mean((2, 3), keepdim=True)
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_bias_ = (None)
        x_113 = torch.sigmoid(x_112)
        x_112 = None
        input_4 = torch.mul(x_110, x_113)
        x_110 = x_113 = None
        x_114 = input_4 + input_3
        input_4 = input_3 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            768,
        )
        x_114 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=False)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.relu(x_121, inplace=False)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.relu(x_124, inplace=False)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.relu(x_127, inplace=False)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_131 = torch.nn.functional.relu(x_130, inplace=False)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.relu(x_133, inplace=False)
        x_133 = None
        x_135 = torch.cat([x_116, x_119, x_122, x_125, x_128, x_131, x_134], dim=1)
        x_116 = x_119 = x_122 = x_125 = x_128 = x_131 = x_134 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=False)
        x_137 = None
        x_139 = x_138.mean((2, 3), keepdim=True)
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_bias_ = (None)
        x_141 = torch.sigmoid(x_140)
        x_140 = None
        input_5 = torch.mul(x_138, x_141)
        x_138 = x_141 = None
        x_142 = torch.nn.functional.adaptive_avg_pool2d(input_5, 1)
        input_5 = None
        input_6 = torch.conv2d(
            x_142,
            l_self_modules_head_modules_last_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = (
            l_self_modules_head_modules_last_conv_modules_0_parameters_weight_
        ) = None
        input_7 = torch.nn.functional.relu(input_6, inplace=False)
        input_6 = None
        x_143 = torch.nn.functional.dropout(input_7, 0.0, False, False)
        input_7 = None
        x_144 = x_143.flatten(1, -1)
        x_143 = None
        x_145 = torch._C._nn.linear(
            x_144,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_144 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_145,)
