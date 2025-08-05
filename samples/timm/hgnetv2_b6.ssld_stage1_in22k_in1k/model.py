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
        L_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_stem4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_stem4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_last_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_
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
        x_3 = torch._C._nn.pad(x_2, (0, 1, 0, 1), "constant", None)
        x_2 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stem_modules_stem2a_modules_conv_parameters_weight_ = None
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = (
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem2a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem2a_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem2a_modules_bn_parameters_bias_ = None
        x_6 = torch.nn.functional.relu(x_5, inplace=False)
        x_5 = None
        x2 = torch._C._nn.pad(x_6, (0, 1, 0, 1), "constant", None)
        x_6 = None
        x_7 = torch.conv2d(
            x2,
            l_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2 = l_self_modules_stem_modules_stem2b_modules_conv_parameters_weight_ = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem2b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem2b_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem2b_modules_bn_parameters_bias_ = None
        x_9 = torch.nn.functional.relu(x_8, inplace=False)
        x_8 = None
        x1 = torch.nn.functional.max_pool2d(
            x_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False
        )
        x_3 = None
        x_10 = torch.cat([x1, x_9], dim=1)
        x1 = x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_stem_modules_stem3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_stem_modules_stem3_modules_conv_parameters_weight_ = None
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem3_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = (
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem3_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem3_modules_bn_parameters_bias_ = None
        x_13 = torch.nn.functional.relu(x_12, inplace=False)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stem_modules_stem4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_stem_modules_stem4_modules_conv_parameters_weight_ = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_stem4_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_stem4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_stem4_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_stem4_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_stem4_modules_bn_parameters_bias_ = None
        x_16 = torch.nn.functional.relu(x_15, inplace=False)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
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
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_19 = torch.nn.functional.relu(x_18, inplace=False)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
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
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=False)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
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
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=False)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
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
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=False)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
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
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=False)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
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
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=False)
        x_33 = None
        x_35 = torch.cat([x_16, x_19, x_22, x_25, x_28, x_31, x_34], dim=1)
        x_16 = x_19 = x_22 = x_25 = x_28 = x_31 = x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=False)
        x_37 = None
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
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.relu(x_43, inplace=False)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=False)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.relu(x_49, inplace=False)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=False)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.relu(x_55, inplace=False)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_59 = torch.nn.functional.relu(x_58, inplace=False)
        x_58 = None
        x_60 = torch.cat([x_41, x_44, x_47, x_50, x_53, x_56, x_59], dim=1)
        x_44 = x_47 = x_50 = x_53 = x_56 = x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_63 = torch.nn.functional.relu(x_62, inplace=False)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_66 = torch.nn.functional.relu(x_65, inplace=False)
        x_65 = None
        x_67 = x_66 + x_41
        x_66 = x_41 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        x_67 = l_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_70 = torch.conv2d(
            x_69,
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
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=False)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
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
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=False)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
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
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
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
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=False)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
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
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=False)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
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
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.relu(x_86, inplace=False)
        x_86 = None
        x_88 = torch.cat([x_69, x_72, x_75, x_78, x_81, x_84, x_87], dim=1)
        x_69 = x_72 = x_75 = x_78 = x_81 = x_84 = x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=False)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=False)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=False)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=False)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=False)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=False)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=False)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=False)
        x_111 = None
        x_113 = torch.cat([x_94, x_97, x_100, x_103, x_106, x_109, x_112], dim=1)
        x_97 = x_100 = x_103 = x_106 = x_109 = x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_116 = torch.nn.functional.relu(x_115, inplace=False)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=False)
        x_118 = None
        x_120 = x_119 + x_94
        x_119 = x_94 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_0_modules_bn_parameters_bias_ = (None)
        x_123 = torch.nn.functional.relu(x_122, inplace=False)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_1_modules_bn_parameters_bias_ = (None)
        x_126 = torch.nn.functional.relu(x_125, inplace=False)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_2_modules_bn_parameters_bias_ = (None)
        x_129 = torch.nn.functional.relu(x_128, inplace=False)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_conv_parameters_weight_ = (
            None
        )
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_3_modules_bn_parameters_bias_ = (None)
        x_132 = torch.nn.functional.relu(x_131, inplace=False)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_conv_parameters_weight_ = (
            None
        )
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_4_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=False)
        x_134 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_conv_parameters_weight_ = (
            None
        )
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_layers_modules_5_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=False)
        x_137 = None
        x_139 = torch.cat([x_120, x_123, x_126, x_129, x_132, x_135, x_138], dim=1)
        x_123 = x_126 = x_129 = x_132 = x_135 = x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=False)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=False)
        x_144 = None
        x_146 = x_145 + x_120
        x_145 = x_120 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            512,
        )
        x_146 = l_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_149 = torch.conv2d(
            x_148,
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
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_150 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=False)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
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
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_155 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=False)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
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
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=False)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=False)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_173 = torch.nn.functional.relu(x_172, inplace=False)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_175 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_178 = torch.nn.functional.relu(x_177, inplace=False)
        x_177 = None
        x_179 = torch.cat([x_148, x_153, x_158, x_163, x_168, x_173, x_178], dim=1)
        x_148 = x_153 = x_158 = x_163 = x_168 = x_173 = x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=False)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=False)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
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
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_187 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=False)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
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
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_192 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_195 = torch.nn.functional.relu(x_194, inplace=False)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
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
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=False)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_202 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_205 = torch.nn.functional.relu(x_204, inplace=False)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_207 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_210 = torch.nn.functional.relu(x_209, inplace=False)
        x_209 = None
        x_211 = torch.conv2d(
            x_210,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_212 = torch.nn.functional.batch_norm(
            x_211,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_211 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_212 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_213 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_215 = torch.nn.functional.relu(x_214, inplace=False)
        x_214 = None
        x_216 = torch.cat([x_185, x_190, x_195, x_200, x_205, x_210, x_215], dim=1)
        x_190 = x_195 = x_200 = x_205 = x_210 = x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_219 = torch.nn.functional.relu(x_218, inplace=False)
        x_218 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_222 = torch.nn.functional.relu(x_221, inplace=False)
        x_221 = None
        x_223 = x_222 + x_185
        x_222 = x_185 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_225 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_228 = torch.nn.functional.relu(x_227, inplace=False)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_230 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_231 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_233 = torch.nn.functional.relu(x_232, inplace=False)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_235 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_238 = torch.nn.functional.relu(x_237, inplace=False)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_243 = torch.nn.functional.relu(x_242, inplace=False)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_245 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_246 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_248 = torch.nn.functional.relu(x_247, inplace=False)
        x_247 = None
        x_249 = torch.conv2d(
            x_248,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_250 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_253 = torch.nn.functional.relu(x_252, inplace=False)
        x_252 = None
        x_254 = torch.cat([x_223, x_228, x_233, x_238, x_243, x_248, x_253], dim=1)
        x_228 = x_233 = x_238 = x_243 = x_248 = x_253 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_254 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_256 = torch.nn.functional.batch_norm(
            x_255,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_255 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_257 = torch.nn.functional.relu(x_256, inplace=False)
        x_256 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_260 = torch.nn.functional.relu(x_259, inplace=False)
        x_259 = None
        x_261 = x_260 + x_223
        x_260 = x_223 = None
        x_262 = torch.conv2d(
            x_261,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_263 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_266 = torch.nn.functional.relu(x_265, inplace=False)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_268 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_269 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_271 = torch.nn.functional.relu(x_270, inplace=False)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_272 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_273 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_276 = torch.nn.functional.relu(x_275, inplace=False)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_279 = torch.conv2d(
            x_278,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_278 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_280 = torch.nn.functional.batch_norm(
            x_279,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_279 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_281 = torch.nn.functional.relu(x_280, inplace=False)
        x_280 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_283 = torch.nn.functional.batch_norm(
            x_282,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_282 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_283 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_284 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_286 = torch.nn.functional.relu(x_285, inplace=False)
        x_285 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_289 = torch.conv2d(
            x_288,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_288 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_290 = torch.nn.functional.batch_norm(
            x_289,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_289 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_291 = torch.nn.functional.relu(x_290, inplace=False)
        x_290 = None
        x_292 = torch.cat([x_261, x_266, x_271, x_276, x_281, x_286, x_291], dim=1)
        x_266 = x_271 = x_276 = x_281 = x_286 = x_291 = None
        x_293 = torch.conv2d(
            x_292,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_292 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_294 = torch.nn.functional.batch_norm(
            x_293,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_293 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_295 = torch.nn.functional.relu(x_294, inplace=False)
        x_294 = None
        x_296 = torch.conv2d(
            x_295,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_295 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_297 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_296 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_298 = torch.nn.functional.relu(x_297, inplace=False)
        x_297 = None
        x_299 = x_298 + x_261
        x_298 = x_261 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_301 = torch.nn.functional.batch_norm(
            x_300,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_300 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_302 = torch.conv2d(
            x_301,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_301 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_303 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_302 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_304 = torch.nn.functional.relu(x_303, inplace=False)
        x_303 = None
        x_305 = torch.conv2d(
            x_304,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_307 = torch.conv2d(
            x_306,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_306 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_308 = torch.nn.functional.batch_norm(
            x_307,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_307 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_309 = torch.nn.functional.relu(x_308, inplace=False)
        x_308 = None
        x_310 = torch.conv2d(
            x_309,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_311 = torch.nn.functional.batch_norm(
            x_310,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_310 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_312 = torch.conv2d(
            x_311,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_311 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_313 = torch.nn.functional.batch_norm(
            x_312,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_312 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_314 = torch.nn.functional.relu(x_313, inplace=False)
        x_313 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_317 = torch.conv2d(
            x_316,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_316 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_317 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_319 = torch.nn.functional.relu(x_318, inplace=False)
        x_318 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_321 = torch.nn.functional.batch_norm(
            x_320,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_320 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_322 = torch.conv2d(
            x_321,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_321 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_323 = torch.nn.functional.batch_norm(
            x_322,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_322 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_324 = torch.nn.functional.relu(x_323, inplace=False)
        x_323 = None
        x_325 = torch.conv2d(
            x_324,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_327 = torch.conv2d(
            x_326,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_326 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_327 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_329 = torch.nn.functional.relu(x_328, inplace=False)
        x_328 = None
        x_330 = torch.cat([x_299, x_304, x_309, x_314, x_319, x_324, x_329], dim=1)
        x_304 = x_309 = x_314 = x_319 = x_324 = x_329 = None
        x_331 = torch.conv2d(
            x_330,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_330 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_332 = torch.nn.functional.batch_norm(
            x_331,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_331 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_333 = torch.nn.functional.relu(x_332, inplace=False)
        x_332 = None
        x_334 = torch.conv2d(
            x_333,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_333 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_335 = torch.nn.functional.batch_norm(
            x_334,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_334 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_336 = torch.nn.functional.relu(x_335, inplace=False)
        x_335 = None
        x_337 = x_336 + x_299
        x_336 = x_299 = None
        x_338 = torch.conv2d(
            x_337,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_340 = torch.conv2d(
            x_339,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_339 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_341 = torch.nn.functional.batch_norm(
            x_340,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_340 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_342 = torch.nn.functional.relu(x_341, inplace=False)
        x_341 = None
        x_343 = torch.conv2d(
            x_342,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_344 = torch.nn.functional.batch_norm(
            x_343,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_343 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_345 = torch.conv2d(
            x_344,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_344 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_346 = torch.nn.functional.batch_norm(
            x_345,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_345 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_347 = torch.nn.functional.relu(x_346, inplace=False)
        x_346 = None
        x_348 = torch.conv2d(
            x_347,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_349 = torch.nn.functional.batch_norm(
            x_348,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_348 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_350 = torch.conv2d(
            x_349,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_349 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_351 = torch.nn.functional.batch_norm(
            x_350,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_350 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_352 = torch.nn.functional.relu(x_351, inplace=False)
        x_351 = None
        x_353 = torch.conv2d(
            x_352,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_354 = torch.nn.functional.batch_norm(
            x_353,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_353 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_355 = torch.conv2d(
            x_354,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_354 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_356 = torch.nn.functional.batch_norm(
            x_355,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_355 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_357 = torch.nn.functional.relu(x_356, inplace=False)
        x_356 = None
        x_358 = torch.conv2d(
            x_357,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_359 = torch.nn.functional.batch_norm(
            x_358,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_358 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_360 = torch.conv2d(
            x_359,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_359 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_361 = torch.nn.functional.batch_norm(
            x_360,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_360 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_362 = torch.nn.functional.relu(x_361, inplace=False)
        x_361 = None
        x_363 = torch.conv2d(
            x_362,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_363 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_365 = torch.conv2d(
            x_364,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_364 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_366 = torch.nn.functional.batch_norm(
            x_365,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_365 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_367 = torch.nn.functional.relu(x_366, inplace=False)
        x_366 = None
        x_368 = torch.cat([x_337, x_342, x_347, x_352, x_357, x_362, x_367], dim=1)
        x_342 = x_347 = x_352 = x_357 = x_362 = x_367 = None
        x_369 = torch.conv2d(
            x_368,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_368 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_371 = torch.nn.functional.relu(x_370, inplace=False)
        x_370 = None
        x_372 = torch.conv2d(
            x_371,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_371 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_373 = torch.nn.functional.batch_norm(
            x_372,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_372 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_374 = torch.nn.functional.relu(x_373, inplace=False)
        x_373 = None
        x_375 = x_374 + x_337
        x_374 = x_337 = None
        x_376 = torch.conv2d(
            x_375,
            l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1024,
        )
        x_375 = l_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_377 = torch.nn.functional.batch_norm(
            x_376,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_376 = l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_378 = torch.conv2d(
            x_377,
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
        x_379 = torch.nn.functional.batch_norm(
            x_378,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_378 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_379 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_381 = torch.nn.functional.batch_norm(
            x_380,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_380 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_382 = torch.nn.functional.relu(x_381, inplace=False)
        x_381 = None
        x_383 = torch.conv2d(
            x_382,
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
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_383 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_385 = torch.conv2d(
            x_384,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_384 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_386 = torch.nn.functional.batch_norm(
            x_385,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_385 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_387 = torch.nn.functional.relu(x_386, inplace=False)
        x_386 = None
        x_388 = torch.conv2d(
            x_387,
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
        x_389 = torch.nn.functional.batch_norm(
            x_388,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_388 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_390 = torch.conv2d(
            x_389,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_389 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_391 = torch.nn.functional.batch_norm(
            x_390,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_390 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_392 = torch.nn.functional.relu(x_391, inplace=False)
        x_391 = None
        x_393 = torch.conv2d(
            x_392,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_394 = torch.nn.functional.batch_norm(
            x_393,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_393 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_395 = torch.conv2d(
            x_394,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_394 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_396 = torch.nn.functional.batch_norm(
            x_395,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_395 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_397 = torch.nn.functional.relu(x_396, inplace=False)
        x_396 = None
        x_398 = torch.conv2d(
            x_397,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_399 = torch.nn.functional.batch_norm(
            x_398,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_398 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_400 = torch.conv2d(
            x_399,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_399 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_401 = torch.nn.functional.batch_norm(
            x_400,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_400 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_402 = torch.nn.functional.relu(x_401, inplace=False)
        x_401 = None
        x_403 = torch.conv2d(
            x_402,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_404 = torch.nn.functional.batch_norm(
            x_403,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_403 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_405 = torch.conv2d(
            x_404,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_404 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_406 = torch.nn.functional.batch_norm(
            x_405,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_405 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_407 = torch.nn.functional.relu(x_406, inplace=False)
        x_406 = None
        x_408 = torch.cat([x_377, x_382, x_387, x_392, x_397, x_402, x_407], dim=1)
        x_377 = x_382 = x_387 = x_392 = x_397 = x_402 = x_407 = None
        x_409 = torch.conv2d(
            x_408,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_408 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_410 = torch.nn.functional.batch_norm(
            x_409,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_409 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_411 = torch.nn.functional.relu(x_410, inplace=False)
        x_410 = None
        x_412 = torch.conv2d(
            x_411,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_411 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_413 = torch.nn.functional.batch_norm(
            x_412,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_412 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_414 = torch.nn.functional.relu(x_413, inplace=False)
        x_413 = None
        x_415 = torch.conv2d(
            x_414,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_416 = torch.nn.functional.batch_norm(
            x_415,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_415 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_417 = torch.conv2d(
            x_416,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_416 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_418 = torch.nn.functional.batch_norm(
            x_417,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_417 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_419 = torch.nn.functional.relu(x_418, inplace=False)
        x_418 = None
        x_420 = torch.conv2d(
            x_419,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_421 = torch.nn.functional.batch_norm(
            x_420,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_420 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_422 = torch.conv2d(
            x_421,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_421 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_423 = torch.nn.functional.batch_norm(
            x_422,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_422 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_424 = torch.nn.functional.relu(x_423, inplace=False)
        x_423 = None
        x_425 = torch.conv2d(
            x_424,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_426 = torch.nn.functional.batch_norm(
            x_425,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_425 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_427 = torch.conv2d(
            x_426,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_426 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_428 = torch.nn.functional.batch_norm(
            x_427,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_427 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_429 = torch.nn.functional.relu(x_428, inplace=False)
        x_428 = None
        x_430 = torch.conv2d(
            x_429,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_431 = torch.nn.functional.batch_norm(
            x_430,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_430 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_432 = torch.conv2d(
            x_431,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_431 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_433 = torch.nn.functional.batch_norm(
            x_432,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_432 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_434 = torch.nn.functional.relu(x_433, inplace=False)
        x_433 = None
        x_435 = torch.conv2d(
            x_434,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_436 = torch.nn.functional.batch_norm(
            x_435,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_435 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_437 = torch.conv2d(
            x_436,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_436 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_438 = torch.nn.functional.batch_norm(
            x_437,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_437 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_439 = torch.nn.functional.relu(x_438, inplace=False)
        x_438 = None
        x_440 = torch.conv2d(
            x_439,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_441 = torch.nn.functional.batch_norm(
            x_440,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_440 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_442 = torch.conv2d(
            x_441,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_441 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_443 = torch.nn.functional.batch_norm(
            x_442,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_442 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_444 = torch.nn.functional.relu(x_443, inplace=False)
        x_443 = None
        x_445 = torch.cat([x_414, x_419, x_424, x_429, x_434, x_439, x_444], dim=1)
        x_419 = x_424 = x_429 = x_434 = x_439 = x_444 = None
        x_446 = torch.conv2d(
            x_445,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_445 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_447 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_446 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_448 = torch.nn.functional.relu(x_447, inplace=False)
        x_447 = None
        x_449 = torch.conv2d(
            x_448,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_448 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_450 = torch.nn.functional.batch_norm(
            x_449,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_449 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_451 = torch.nn.functional.relu(x_450, inplace=False)
        x_450 = None
        x_452 = x_451 + x_414
        x_451 = x_414 = None
        x_453 = torch.conv2d(
            x_452,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_454 = torch.nn.functional.batch_norm(
            x_453,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_453 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_455 = torch.conv2d(
            x_454,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_454 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_456 = torch.nn.functional.batch_norm(
            x_455,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_455 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_457 = torch.nn.functional.relu(x_456, inplace=False)
        x_456 = None
        x_458 = torch.conv2d(
            x_457,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_459 = torch.nn.functional.batch_norm(
            x_458,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_458 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_460 = torch.conv2d(
            x_459,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_459 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_461 = torch.nn.functional.batch_norm(
            x_460,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_460 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_462 = torch.nn.functional.relu(x_461, inplace=False)
        x_461 = None
        x_463 = torch.conv2d(
            x_462,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_464 = torch.nn.functional.batch_norm(
            x_463,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_463 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_465 = torch.conv2d(
            x_464,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_464 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_466 = torch.nn.functional.batch_norm(
            x_465,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_465 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_467 = torch.nn.functional.relu(x_466, inplace=False)
        x_466 = None
        x_468 = torch.conv2d(
            x_467,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_469 = torch.nn.functional.batch_norm(
            x_468,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_468 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_470 = torch.conv2d(
            x_469,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_469 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_471 = torch.nn.functional.batch_norm(
            x_470,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_470 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_472 = torch.nn.functional.relu(x_471, inplace=False)
        x_471 = None
        x_473 = torch.conv2d(
            x_472,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_474 = torch.nn.functional.batch_norm(
            x_473,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_473 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_475 = torch.conv2d(
            x_474,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_474 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_476 = torch.nn.functional.batch_norm(
            x_475,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_475 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_477 = torch.nn.functional.relu(x_476, inplace=False)
        x_476 = None
        x_478 = torch.conv2d(
            x_477,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_479 = torch.nn.functional.batch_norm(
            x_478,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_478 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_480 = torch.conv2d(
            x_479,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_479 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_481 = torch.nn.functional.batch_norm(
            x_480,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_480 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_layers_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_482 = torch.nn.functional.relu(x_481, inplace=False)
        x_481 = None
        x_483 = torch.cat([x_452, x_457, x_462, x_467, x_472, x_477, x_482], dim=1)
        x_457 = x_462 = x_467 = x_472 = x_477 = x_482 = None
        x_484 = torch.conv2d(
            x_483,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_483 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_conv_parameters_weight_ = (None)
        x_485 = torch.nn.functional.batch_norm(
            x_484,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_484 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_0_modules_bn_parameters_bias_ = (None)
        x_486 = torch.nn.functional.relu(x_485, inplace=False)
        x_485 = None
        x_487 = torch.conv2d(
            x_486,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_486 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_conv_parameters_weight_ = (None)
        x_488 = torch.nn.functional.batch_norm(
            x_487,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_487 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_aggregation_modules_1_modules_bn_parameters_bias_ = (None)
        x_489 = torch.nn.functional.relu(x_488, inplace=False)
        x_488 = None
        x_490 = x_489 + x_452
        x_489 = x_452 = None
        x_491 = torch.nn.functional.adaptive_avg_pool2d(x_490, 1)
        x_490 = None
        input_1 = torch.conv2d(
            x_491,
            l_self_modules_head_modules_last_conv_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_491 = (
            l_self_modules_head_modules_last_conv_modules_0_parameters_weight_
        ) = None
        input_2 = torch.nn.functional.relu(input_1, inplace=False)
        input_1 = None
        x_492 = torch.nn.functional.dropout(input_2, 0.0, False, False)
        input_2 = None
        x_493 = x_492.flatten(1, -1)
        x_492 = None
        x_494 = torch._C._nn.linear(
            x_493,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_493 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_494,)
