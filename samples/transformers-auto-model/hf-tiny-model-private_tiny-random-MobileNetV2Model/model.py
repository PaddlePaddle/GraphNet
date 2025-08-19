import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s99: torch.SymInt,
        L_pixel_values_: torch.Tensor,
        L_self_modules_conv_stem_modules_first_conv_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_first_conv_modules_normalization_momentum: torch.Tensor,
        L_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_first_conv_modules_normalization_eps: torch.Tensor,
        L_self_modules_conv_stem_modules_first_conv_modules_activation_min_val: torch.Tensor,
        L_self_modules_conv_stem_modules_first_conv_modules_activation_max_val: torch.Tensor,
        L_self_modules_conv_stem_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_conv_stem_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_conv_stem_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_conv_1x1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_1x1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_conv_1x1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_1x1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_1x1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_1x1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_1x1_modules_normalization_eps: torch.Tensor,
        L_self_modules_conv_1x1_modules_activation_min_val: torch.Tensor,
        L_self_modules_conv_1x1_modules_activation_max_val: torch.Tensor,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_conv_stem_modules_first_conv_modules_convolution_parameters_weight_ = L_self_modules_conv_stem_modules_first_conv_modules_convolution_parameters_weight_
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_momentum = (
            L_self_modules_conv_stem_modules_first_conv_modules_normalization_momentum
        )
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_mean_ = L_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_mean_
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_var_ = L_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_var_
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_weight_ = L_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_weight_
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_bias_ = L_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_bias_
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_eps = (
            L_self_modules_conv_stem_modules_first_conv_modules_normalization_eps
        )
        l_self_modules_conv_stem_modules_first_conv_modules_activation_min_val = (
            L_self_modules_conv_stem_modules_first_conv_modules_activation_min_val
        )
        l_self_modules_conv_stem_modules_first_conv_modules_activation_max_val = (
            L_self_modules_conv_stem_modules_first_conv_modules_activation_max_val
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_conv_stem_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_momentum = (
            L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_momentum
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_conv_stem_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_conv_stem_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_conv_stem_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_conv_stem_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_momentum = (
            L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_momentum
        )
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_0_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_0_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_0_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_1_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_1_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_1_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_2_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_2_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_2_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_3_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_3_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_3_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_4_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_4_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_4_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_5_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_5_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_5_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_6_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_6_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_6_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_7_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_7_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_7_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_8_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_8_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_8_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_9_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_min_val = (
            L_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_min_val
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_max_val = (
            L_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_max_val
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_9_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_9_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_10_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_min_val = L_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_min_val
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_max_val = L_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_max_val
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_10_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_10_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_11_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_min_val = L_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_min_val
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_max_val = L_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_max_val
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_11_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_11_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_12_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_min_val = L_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_min_val
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_max_val = L_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_max_val
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_12_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_12_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_13_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_min_val = L_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_min_val
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_max_val = L_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_max_val
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_13_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_13_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_14_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_min_val = L_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_min_val
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_max_val = L_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_max_val
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_14_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_14_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_15_modules_expand_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_momentum = L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_eps
        )
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_min_val = L_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_min_val
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_max_val = L_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_max_val
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_15_modules_conv_3x3_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_momentum = L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_momentum
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_eps = (
            L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_eps
        )
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_min_val = (
            L_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_min_val
        )
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_max_val = (
            L_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_max_val
        )
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_convolution_parameters_weight_ = L_self_modules_layer_modules_15_modules_reduce_1x1_modules_convolution_parameters_weight_
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_momentum = L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_momentum
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_mean_
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_var_ = L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_var_
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_weight_ = L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_weight_
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_bias_ = L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_bias_
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_eps = (
            L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_eps
        )
        l_self_modules_conv_1x1_modules_convolution_parameters_weight_ = (
            L_self_modules_conv_1x1_modules_convolution_parameters_weight_
        )
        l_self_modules_conv_1x1_modules_normalization_momentum = (
            L_self_modules_conv_1x1_modules_normalization_momentum
        )
        l_self_modules_conv_1x1_modules_normalization_buffers_running_mean_ = (
            L_self_modules_conv_1x1_modules_normalization_buffers_running_mean_
        )
        l_self_modules_conv_1x1_modules_normalization_buffers_running_var_ = (
            L_self_modules_conv_1x1_modules_normalization_buffers_running_var_
        )
        l_self_modules_conv_1x1_modules_normalization_parameters_weight_ = (
            L_self_modules_conv_1x1_modules_normalization_parameters_weight_
        )
        l_self_modules_conv_1x1_modules_normalization_parameters_bias_ = (
            L_self_modules_conv_1x1_modules_normalization_parameters_bias_
        )
        l_self_modules_conv_1x1_modules_normalization_eps = (
            L_self_modules_conv_1x1_modules_normalization_eps
        )
        l_self_modules_conv_1x1_modules_activation_min_val = (
            L_self_modules_conv_1x1_modules_activation_min_val
        )
        l_self_modules_conv_1x1_modules_activation_max_val = (
            L_self_modules_conv_1x1_modules_activation_max_val
        )
        features = torch._C._nn.pad(l_pixel_values_, (0, 1, 0, 1), "constant", 0.0)
        l_pixel_values_ = None
        features_1 = torch.conv2d(
            features,
            l_self_modules_conv_stem_modules_first_conv_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        features = l_self_modules_conv_stem_modules_first_conv_modules_convolution_parameters_weight_ = (None)
        item = (
            l_self_modules_conv_stem_modules_first_conv_modules_normalization_momentum.item()
        )
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_momentum = (
            None
        )
        item_1 = (
            l_self_modules_conv_stem_modules_first_conv_modules_normalization_eps.item()
        )
        l_self_modules_conv_stem_modules_first_conv_modules_normalization_eps = None
        features_2 = torch.nn.functional.batch_norm(
            features_1,
            l_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_mean_,
            l_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_var_,
            l_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_weight_,
            l_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_bias_,
            False,
            item,
            item_1,
        )
        features_1 = l_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_mean_ = l_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_var_ = l_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_weight_ = l_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_bias_ = (item) = (
            item_1
        ) = None
        item_2 = (
            l_self_modules_conv_stem_modules_first_conv_modules_activation_min_val.item()
        )
        l_self_modules_conv_stem_modules_first_conv_modules_activation_min_val = None
        item_3 = (
            l_self_modules_conv_stem_modules_first_conv_modules_activation_max_val.item()
        )
        l_self_modules_conv_stem_modules_first_conv_modules_activation_max_val = None
        features_3 = torch.nn.functional.hardtanh(features_2, item_2, item_3, False)
        features_2 = item_2 = item_3 = None
        features_4 = torch._C._nn.pad(features_3, (1, 1, 1, 1), "constant", 0.0)
        features_3 = None
        features_5 = torch.conv2d(
            features_4,
            l_self_modules_conv_stem_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            8,
        )
        features_4 = l_self_modules_conv_stem_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_4 = (
            l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_momentum = None
        item_5 = (
            l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_eps = None
        features_6 = torch.nn.functional.batch_norm(
            features_5,
            l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_4,
            item_5,
        )
        features_5 = l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_4) = (
            item_5
        ) = None
        item_6 = (
            l_self_modules_conv_stem_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_activation_min_val = None
        item_7 = (
            l_self_modules_conv_stem_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_conv_stem_modules_conv_3x3_modules_activation_max_val = None
        features_7 = torch.nn.functional.hardtanh(features_6, item_6, item_7, False)
        features_6 = item_6 = item_7 = None
        features_8 = torch._C._nn.pad(features_7, (0, 0, 0, 0), "constant", 0.0)
        features_7 = None
        features_9 = torch.conv2d(
            features_8,
            l_self_modules_conv_stem_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_8 = l_self_modules_conv_stem_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_8 = (
            l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_9 = (
            l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_eps = None
        features_10 = torch.nn.functional.batch_norm(
            features_9,
            l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_8,
            item_9,
        )
        features_9 = l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_8) = (
            item_9
        ) = None
        features_11 = torch._C._nn.pad(features_10, (0, 0, 0, 0), "constant", 0.0)
        features_10 = None
        features_12 = torch.conv2d(
            features_11,
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_11 = l_self_modules_layer_modules_0_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_10 = (
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_11 = (
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_13 = torch.nn.functional.batch_norm(
            features_12,
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_10,
            item_11,
        )
        features_12 = l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_10) = (
            item_11
        ) = None
        item_12 = (
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_13 = (
            l_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_14 = torch.nn.functional.hardtanh(features_13, item_12, item_13, False)
        features_13 = item_12 = item_13 = None
        features_15 = torch._C._nn.pad(features_14, (0, 1, 0, 1), "constant", 0.0)
        features_14 = None
        features_16 = torch.conv2d(
            features_15,
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            48,
        )
        features_15 = l_self_modules_layer_modules_0_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_14 = (
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_15 = (
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_eps = None
        features_17 = torch.nn.functional.batch_norm(
            features_16,
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_14,
            item_15,
        )
        features_16 = l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_14) = (
            item_15
        ) = None
        item_16 = (
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_17 = (
            l_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_18 = torch.nn.functional.hardtanh(features_17, item_16, item_17, False)
        features_17 = item_16 = item_17 = None
        features_19 = torch._C._nn.pad(features_18, (0, 0, 0, 0), "constant", 0.0)
        features_18 = None
        features_20 = torch.conv2d(
            features_19,
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_19 = l_self_modules_layer_modules_0_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_18 = (
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_19 = (
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_21 = torch.nn.functional.batch_norm(
            features_20,
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_18,
            item_19,
        )
        features_20 = l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_18) = (
            item_19
        ) = None
        features_22 = torch._C._nn.pad(features_21, (0, 0, 0, 0), "constant", 0.0)
        features_23 = torch.conv2d(
            features_22,
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_22 = l_self_modules_layer_modules_1_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_20 = (
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_21 = (
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_24 = torch.nn.functional.batch_norm(
            features_23,
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_20,
            item_21,
        )
        features_23 = l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_20) = (
            item_21
        ) = None
        item_22 = (
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_23 = (
            l_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_25 = torch.nn.functional.hardtanh(features_24, item_22, item_23, False)
        features_24 = item_22 = item_23 = None
        features_26 = torch._C._nn.pad(features_25, (1, 1, 1, 1), "constant", 0.0)
        features_25 = None
        features_27 = torch.conv2d(
            features_26,
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            48,
        )
        features_26 = l_self_modules_layer_modules_1_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_24 = (
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_25 = (
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_eps = None
        features_28 = torch.nn.functional.batch_norm(
            features_27,
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        features_27 = l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_24) = (
            item_25
        ) = None
        item_26 = (
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_27 = (
            l_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_29 = torch.nn.functional.hardtanh(features_28, item_26, item_27, False)
        features_28 = item_26 = item_27 = None
        features_30 = torch._C._nn.pad(features_29, (0, 0, 0, 0), "constant", 0.0)
        features_29 = None
        features_31 = torch.conv2d(
            features_30,
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_30 = l_self_modules_layer_modules_1_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_28 = (
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_29 = (
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_32 = torch.nn.functional.batch_norm(
            features_31,
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_28,
            item_29,
        )
        features_31 = l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_28) = (
            item_29
        ) = None
        hidden_states = features_21 + features_32
        features_21 = features_32 = None
        features_33 = torch._C._nn.pad(hidden_states, (0, 0, 0, 0), "constant", 0.0)
        hidden_states = None
        features_34 = torch.conv2d(
            features_33,
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_33 = l_self_modules_layer_modules_2_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_30 = (
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_31 = (
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_35 = torch.nn.functional.batch_norm(
            features_34,
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_30,
            item_31,
        )
        features_34 = l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_30) = (
            item_31
        ) = None
        item_32 = (
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_33 = (
            l_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_36 = torch.nn.functional.hardtanh(features_35, item_32, item_33, False)
        features_35 = item_32 = item_33 = None
        features_37 = torch._C._nn.pad(features_36, (0, 1, 0, 1), "constant", 0.0)
        features_36 = None
        features_38 = torch.conv2d(
            features_37,
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            48,
        )
        features_37 = l_self_modules_layer_modules_2_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_34 = (
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_35 = (
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_eps = None
        features_39 = torch.nn.functional.batch_norm(
            features_38,
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_34,
            item_35,
        )
        features_38 = l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_34) = (
            item_35
        ) = None
        item_36 = (
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_37 = (
            l_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_40 = torch.nn.functional.hardtanh(features_39, item_36, item_37, False)
        features_39 = item_36 = item_37 = None
        features_41 = torch._C._nn.pad(features_40, (0, 0, 0, 0), "constant", 0.0)
        features_40 = None
        features_42 = torch.conv2d(
            features_41,
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_41 = l_self_modules_layer_modules_2_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_38 = (
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_39 = (
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_43 = torch.nn.functional.batch_norm(
            features_42,
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_38,
            item_39,
        )
        features_42 = l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_38) = (
            item_39
        ) = None
        features_44 = torch._C._nn.pad(features_43, (0, 0, 0, 0), "constant", 0.0)
        features_45 = torch.conv2d(
            features_44,
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_44 = l_self_modules_layer_modules_3_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_40 = (
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_41 = (
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_46 = torch.nn.functional.batch_norm(
            features_45,
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_40,
            item_41,
        )
        features_45 = l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_40) = (
            item_41
        ) = None
        item_42 = (
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_43 = (
            l_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_47 = torch.nn.functional.hardtanh(features_46, item_42, item_43, False)
        features_46 = item_42 = item_43 = None
        features_48 = torch._C._nn.pad(features_47, (1, 1, 1, 1), "constant", 0.0)
        features_47 = None
        features_49 = torch.conv2d(
            features_48,
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            48,
        )
        features_48 = l_self_modules_layer_modules_3_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_44 = (
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_45 = (
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_eps = None
        features_50 = torch.nn.functional.batch_norm(
            features_49,
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_44,
            item_45,
        )
        features_49 = l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_44) = (
            item_45
        ) = None
        item_46 = (
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_47 = (
            l_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_51 = torch.nn.functional.hardtanh(features_50, item_46, item_47, False)
        features_50 = item_46 = item_47 = None
        features_52 = torch._C._nn.pad(features_51, (0, 0, 0, 0), "constant", 0.0)
        features_51 = None
        features_53 = torch.conv2d(
            features_52,
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_52 = l_self_modules_layer_modules_3_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_48 = (
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_49 = (
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_54 = torch.nn.functional.batch_norm(
            features_53,
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_48,
            item_49,
        )
        features_53 = l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_48) = (
            item_49
        ) = None
        hidden_states_1 = features_43 + features_54
        features_43 = features_54 = None
        features_55 = torch._C._nn.pad(hidden_states_1, (0, 0, 0, 0), "constant", 0.0)
        features_56 = torch.conv2d(
            features_55,
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_55 = l_self_modules_layer_modules_4_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_50 = (
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_51 = (
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_57 = torch.nn.functional.batch_norm(
            features_56,
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_50,
            item_51,
        )
        features_56 = l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_50) = (
            item_51
        ) = None
        item_52 = (
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_53 = (
            l_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_58 = torch.nn.functional.hardtanh(features_57, item_52, item_53, False)
        features_57 = item_52 = item_53 = None
        features_59 = torch._C._nn.pad(features_58, (1, 1, 1, 1), "constant", 0.0)
        features_58 = None
        features_60 = torch.conv2d(
            features_59,
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            48,
        )
        features_59 = l_self_modules_layer_modules_4_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_54 = (
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_55 = (
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_eps = None
        features_61 = torch.nn.functional.batch_norm(
            features_60,
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_54,
            item_55,
        )
        features_60 = l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_54) = (
            item_55
        ) = None
        item_56 = (
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_57 = (
            l_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_62 = torch.nn.functional.hardtanh(features_61, item_56, item_57, False)
        features_61 = item_56 = item_57 = None
        features_63 = torch._C._nn.pad(features_62, (0, 0, 0, 0), "constant", 0.0)
        features_62 = None
        features_64 = torch.conv2d(
            features_63,
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_63 = l_self_modules_layer_modules_4_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_58 = (
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_59 = (
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_65 = torch.nn.functional.batch_norm(
            features_64,
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_58,
            item_59,
        )
        features_64 = l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_58) = (
            item_59
        ) = None
        hidden_states_2 = hidden_states_1 + features_65
        hidden_states_1 = features_65 = None
        features_66 = torch._C._nn.pad(hidden_states_2, (0, 0, 0, 0), "constant", 0.0)
        hidden_states_2 = None
        features_67 = torch.conv2d(
            features_66,
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_66 = l_self_modules_layer_modules_5_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_60 = (
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_61 = (
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_68 = torch.nn.functional.batch_norm(
            features_67,
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_60,
            item_61,
        )
        features_67 = l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_60) = (
            item_61
        ) = None
        item_62 = (
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_63 = (
            l_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_69 = torch.nn.functional.hardtanh(features_68, item_62, item_63, False)
        features_68 = item_62 = item_63 = None
        features_70 = torch._C._nn.pad(features_69, (0, 1, 0, 1), "constant", 0.0)
        features_69 = None
        features_71 = torch.conv2d(
            features_70,
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            48,
        )
        features_70 = l_self_modules_layer_modules_5_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_64 = (
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_65 = (
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_eps = None
        features_72 = torch.nn.functional.batch_norm(
            features_71,
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_64,
            item_65,
        )
        features_71 = l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_64) = (
            item_65
        ) = None
        item_66 = (
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_67 = (
            l_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_73 = torch.nn.functional.hardtanh(features_72, item_66, item_67, False)
        features_72 = item_66 = item_67 = None
        features_74 = torch._C._nn.pad(features_73, (0, 0, 0, 0), "constant", 0.0)
        features_73 = None
        features_75 = torch.conv2d(
            features_74,
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_74 = l_self_modules_layer_modules_5_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_68 = (
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_69 = (
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_76 = torch.nn.functional.batch_norm(
            features_75,
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_68,
            item_69,
        )
        features_75 = l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_68) = (
            item_69
        ) = None
        features_77 = torch._C._nn.pad(features_76, (0, 0, 0, 0), "constant", 0.0)
        features_78 = torch.conv2d(
            features_77,
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_77 = l_self_modules_layer_modules_6_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_70 = (
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_71 = (
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_79 = torch.nn.functional.batch_norm(
            features_78,
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_70,
            item_71,
        )
        features_78 = l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_70) = (
            item_71
        ) = None
        item_72 = (
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_73 = (
            l_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_80 = torch.nn.functional.hardtanh(features_79, item_72, item_73, False)
        features_79 = item_72 = item_73 = None
        features_81 = torch._C._nn.pad(features_80, (1, 1, 1, 1), "constant", 0.0)
        features_80 = None
        features_82 = torch.conv2d(
            features_81,
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            96,
        )
        features_81 = l_self_modules_layer_modules_6_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_74 = (
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_75 = (
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_eps = None
        features_83 = torch.nn.functional.batch_norm(
            features_82,
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_74,
            item_75,
        )
        features_82 = l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_74) = (
            item_75
        ) = None
        item_76 = (
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_77 = (
            l_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_84 = torch.nn.functional.hardtanh(features_83, item_76, item_77, False)
        features_83 = item_76 = item_77 = None
        features_85 = torch._C._nn.pad(features_84, (0, 0, 0, 0), "constant", 0.0)
        features_84 = None
        features_86 = torch.conv2d(
            features_85,
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_85 = l_self_modules_layer_modules_6_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_78 = (
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_79 = (
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_87 = torch.nn.functional.batch_norm(
            features_86,
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_78,
            item_79,
        )
        features_86 = l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_78) = (
            item_79
        ) = None
        hidden_states_3 = features_76 + features_87
        features_76 = features_87 = None
        features_88 = torch._C._nn.pad(hidden_states_3, (0, 0, 0, 0), "constant", 0.0)
        features_89 = torch.conv2d(
            features_88,
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_88 = l_self_modules_layer_modules_7_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_80 = (
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_81 = (
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_90 = torch.nn.functional.batch_norm(
            features_89,
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_80,
            item_81,
        )
        features_89 = l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_80) = (
            item_81
        ) = None
        item_82 = (
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_83 = (
            l_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_91 = torch.nn.functional.hardtanh(features_90, item_82, item_83, False)
        features_90 = item_82 = item_83 = None
        features_92 = torch._C._nn.pad(features_91, (1, 1, 1, 1), "constant", 0.0)
        features_91 = None
        features_93 = torch.conv2d(
            features_92,
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            96,
        )
        features_92 = l_self_modules_layer_modules_7_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_84 = (
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_85 = (
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_eps = None
        features_94 = torch.nn.functional.batch_norm(
            features_93,
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_84,
            item_85,
        )
        features_93 = l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_84) = (
            item_85
        ) = None
        item_86 = (
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_87 = (
            l_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_95 = torch.nn.functional.hardtanh(features_94, item_86, item_87, False)
        features_94 = item_86 = item_87 = None
        features_96 = torch._C._nn.pad(features_95, (0, 0, 0, 0), "constant", 0.0)
        features_95 = None
        features_97 = torch.conv2d(
            features_96,
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_96 = l_self_modules_layer_modules_7_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_88 = (
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_89 = (
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_98 = torch.nn.functional.batch_norm(
            features_97,
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_88,
            item_89,
        )
        features_97 = l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_88) = (
            item_89
        ) = None
        hidden_states_4 = hidden_states_3 + features_98
        hidden_states_3 = features_98 = None
        features_99 = torch._C._nn.pad(hidden_states_4, (0, 0, 0, 0), "constant", 0.0)
        features_100 = torch.conv2d(
            features_99,
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_99 = l_self_modules_layer_modules_8_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_90 = (
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_91 = (
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_101 = torch.nn.functional.batch_norm(
            features_100,
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_90,
            item_91,
        )
        features_100 = l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_90) = (
            item_91
        ) = None
        item_92 = (
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_93 = (
            l_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_102 = torch.nn.functional.hardtanh(
            features_101, item_92, item_93, False
        )
        features_101 = item_92 = item_93 = None
        features_103 = torch._C._nn.pad(features_102, (1, 1, 1, 1), "constant", 0.0)
        features_102 = None
        features_104 = torch.conv2d(
            features_103,
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            96,
        )
        features_103 = l_self_modules_layer_modules_8_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_94 = (
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_95 = (
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_eps = None
        features_105 = torch.nn.functional.batch_norm(
            features_104,
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_94,
            item_95,
        )
        features_104 = l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_94) = (
            item_95
        ) = None
        item_96 = (
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_97 = (
            l_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_106 = torch.nn.functional.hardtanh(
            features_105, item_96, item_97, False
        )
        features_105 = item_96 = item_97 = None
        features_107 = torch._C._nn.pad(features_106, (0, 0, 0, 0), "constant", 0.0)
        features_106 = None
        features_108 = torch.conv2d(
            features_107,
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_107 = l_self_modules_layer_modules_8_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_98 = (
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_99 = (
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_109 = torch.nn.functional.batch_norm(
            features_108,
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_98,
            item_99,
        )
        features_108 = l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_98) = (
            item_99
        ) = None
        hidden_states_5 = hidden_states_4 + features_109
        hidden_states_4 = features_109 = None
        features_110 = torch._C._nn.pad(hidden_states_5, (0, 0, 0, 0), "constant", 0.0)
        hidden_states_5 = None
        features_111 = torch.conv2d(
            features_110,
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_110 = l_self_modules_layer_modules_9_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_100 = (
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_101 = (
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_112 = torch.nn.functional.batch_norm(
            features_111,
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_100,
            item_101,
        )
        features_111 = l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_100) = (
            item_101
        ) = None
        item_102 = (
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_103 = (
            l_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_113 = torch.nn.functional.hardtanh(
            features_112, item_102, item_103, False
        )
        features_112 = item_102 = item_103 = None
        features_114 = torch._C._nn.pad(features_113, (1, 1, 1, 1), "constant", 0.0)
        features_113 = None
        features_115 = torch.conv2d(
            features_114,
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            96,
        )
        features_114 = l_self_modules_layer_modules_9_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_104 = (
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_105 = (
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_eps = None
        features_116 = torch.nn.functional.batch_norm(
            features_115,
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_104,
            item_105,
        )
        features_115 = l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_104) = (
            item_105
        ) = None
        item_106 = (
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_107 = (
            l_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_117 = torch.nn.functional.hardtanh(
            features_116, item_106, item_107, False
        )
        features_116 = item_106 = item_107 = None
        features_118 = torch._C._nn.pad(features_117, (0, 0, 0, 0), "constant", 0.0)
        features_117 = None
        features_119 = torch.conv2d(
            features_118,
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_118 = l_self_modules_layer_modules_9_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_108 = (
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_109 = (
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_120 = torch.nn.functional.batch_norm(
            features_119,
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_108,
            item_109,
        )
        features_119 = l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_108) = (
            item_109
        ) = None
        features_121 = torch._C._nn.pad(features_120, (0, 0, 0, 0), "constant", 0.0)
        features_122 = torch.conv2d(
            features_121,
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_121 = l_self_modules_layer_modules_10_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_110 = (
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_111 = (
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_123 = torch.nn.functional.batch_norm(
            features_122,
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_110,
            item_111,
        )
        features_122 = l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_110) = (
            item_111
        ) = None
        item_112 = (
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_113 = (
            l_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_124 = torch.nn.functional.hardtanh(
            features_123, item_112, item_113, False
        )
        features_123 = item_112 = item_113 = None
        features_125 = torch._C._nn.pad(features_124, (1, 1, 1, 1), "constant", 0.0)
        features_124 = None
        features_126 = torch.conv2d(
            features_125,
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            144,
        )
        features_125 = l_self_modules_layer_modules_10_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_114 = (
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_115 = (
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_eps = (
            None
        )
        features_127 = torch.nn.functional.batch_norm(
            features_126,
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_114,
            item_115,
        )
        features_126 = l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_114) = (
            item_115
        ) = None
        item_116 = (
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_117 = (
            l_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_128 = torch.nn.functional.hardtanh(
            features_127, item_116, item_117, False
        )
        features_127 = item_116 = item_117 = None
        features_129 = torch._C._nn.pad(features_128, (0, 0, 0, 0), "constant", 0.0)
        features_128 = None
        features_130 = torch.conv2d(
            features_129,
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_129 = l_self_modules_layer_modules_10_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_118 = (
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_119 = (
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_131 = torch.nn.functional.batch_norm(
            features_130,
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_118,
            item_119,
        )
        features_130 = l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_118) = (
            item_119
        ) = None
        hidden_states_6 = features_120 + features_131
        features_120 = features_131 = None
        features_132 = torch._C._nn.pad(hidden_states_6, (0, 0, 0, 0), "constant", 0.0)
        features_133 = torch.conv2d(
            features_132,
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_132 = l_self_modules_layer_modules_11_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_120 = (
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_121 = (
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_134 = torch.nn.functional.batch_norm(
            features_133,
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_120,
            item_121,
        )
        features_133 = l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_120) = (
            item_121
        ) = None
        item_122 = (
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_123 = (
            l_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_135 = torch.nn.functional.hardtanh(
            features_134, item_122, item_123, False
        )
        features_134 = item_122 = item_123 = None
        features_136 = torch._C._nn.pad(features_135, (1, 1, 1, 1), "constant", 0.0)
        features_135 = None
        features_137 = torch.conv2d(
            features_136,
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            144,
        )
        features_136 = l_self_modules_layer_modules_11_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_124 = (
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_125 = (
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_eps = (
            None
        )
        features_138 = torch.nn.functional.batch_norm(
            features_137,
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_124,
            item_125,
        )
        features_137 = l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_124) = (
            item_125
        ) = None
        item_126 = (
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_127 = (
            l_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_139 = torch.nn.functional.hardtanh(
            features_138, item_126, item_127, False
        )
        features_138 = item_126 = item_127 = None
        features_140 = torch._C._nn.pad(features_139, (0, 0, 0, 0), "constant", 0.0)
        features_139 = None
        features_141 = torch.conv2d(
            features_140,
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_140 = l_self_modules_layer_modules_11_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_128 = (
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_129 = (
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_142 = torch.nn.functional.batch_norm(
            features_141,
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_128,
            item_129,
        )
        features_141 = l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_128) = (
            item_129
        ) = None
        hidden_states_7 = hidden_states_6 + features_142
        hidden_states_6 = features_142 = None
        features_143 = torch._C._nn.pad(hidden_states_7, (0, 0, 0, 0), "constant", 0.0)
        hidden_states_7 = None
        features_144 = torch.conv2d(
            features_143,
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_143 = l_self_modules_layer_modules_12_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_130 = (
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_131 = (
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_145 = torch.nn.functional.batch_norm(
            features_144,
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_130,
            item_131,
        )
        features_144 = l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_130) = (
            item_131
        ) = None
        item_132 = (
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_133 = (
            l_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_146 = torch.nn.functional.hardtanh(
            features_145, item_132, item_133, False
        )
        features_145 = item_132 = item_133 = None
        features_147 = torch._C._nn.pad(features_146, (0, 1, 0, 1), "constant", 0.0)
        features_146 = None
        features_148 = torch.conv2d(
            features_147,
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            144,
        )
        features_147 = l_self_modules_layer_modules_12_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_134 = (
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_135 = (
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_eps = (
            None
        )
        features_149 = torch.nn.functional.batch_norm(
            features_148,
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_134,
            item_135,
        )
        features_148 = l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_134) = (
            item_135
        ) = None
        item_136 = (
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_137 = (
            l_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_150 = torch.nn.functional.hardtanh(
            features_149, item_136, item_137, False
        )
        features_149 = item_136 = item_137 = None
        features_151 = torch._C._nn.pad(features_150, (0, 0, 0, 0), "constant", 0.0)
        features_150 = None
        features_152 = torch.conv2d(
            features_151,
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_151 = l_self_modules_layer_modules_12_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_138 = (
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_139 = (
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_153 = torch.nn.functional.batch_norm(
            features_152,
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_138,
            item_139,
        )
        features_152 = l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_138) = (
            item_139
        ) = None
        features_154 = torch._C._nn.pad(features_153, (0, 0, 0, 0), "constant", 0.0)
        features_155 = torch.conv2d(
            features_154,
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_154 = l_self_modules_layer_modules_13_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_140 = (
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_141 = (
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_156 = torch.nn.functional.batch_norm(
            features_155,
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_140,
            item_141,
        )
        features_155 = l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_140) = (
            item_141
        ) = None
        item_142 = (
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_143 = (
            l_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_157 = torch.nn.functional.hardtanh(
            features_156, item_142, item_143, False
        )
        features_156 = item_142 = item_143 = None
        features_158 = torch._C._nn.pad(features_157, (1, 1, 1, 1), "constant", 0.0)
        features_157 = None
        features_159 = torch.conv2d(
            features_158,
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            240,
        )
        features_158 = l_self_modules_layer_modules_13_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_144 = (
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_145 = (
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_eps = (
            None
        )
        features_160 = torch.nn.functional.batch_norm(
            features_159,
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_144,
            item_145,
        )
        features_159 = l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_144) = (
            item_145
        ) = None
        item_146 = (
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_147 = (
            l_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_161 = torch.nn.functional.hardtanh(
            features_160, item_146, item_147, False
        )
        features_160 = item_146 = item_147 = None
        features_162 = torch._C._nn.pad(features_161, (0, 0, 0, 0), "constant", 0.0)
        features_161 = None
        features_163 = torch.conv2d(
            features_162,
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_162 = l_self_modules_layer_modules_13_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_148 = (
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_149 = (
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_164 = torch.nn.functional.batch_norm(
            features_163,
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_148,
            item_149,
        )
        features_163 = l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_148) = (
            item_149
        ) = None
        hidden_states_8 = features_153 + features_164
        features_153 = features_164 = None
        features_165 = torch._C._nn.pad(hidden_states_8, (0, 0, 0, 0), "constant", 0.0)
        features_166 = torch.conv2d(
            features_165,
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_165 = l_self_modules_layer_modules_14_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_150 = (
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_151 = (
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_167 = torch.nn.functional.batch_norm(
            features_166,
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_150,
            item_151,
        )
        features_166 = l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_150) = (
            item_151
        ) = None
        item_152 = (
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_153 = (
            l_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_168 = torch.nn.functional.hardtanh(
            features_167, item_152, item_153, False
        )
        features_167 = item_152 = item_153 = None
        features_169 = torch._C._nn.pad(features_168, (1, 1, 1, 1), "constant", 0.0)
        features_168 = None
        features_170 = torch.conv2d(
            features_169,
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            240,
        )
        features_169 = l_self_modules_layer_modules_14_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_154 = (
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_155 = (
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_eps = (
            None
        )
        features_171 = torch.nn.functional.batch_norm(
            features_170,
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_154,
            item_155,
        )
        features_170 = l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_154) = (
            item_155
        ) = None
        item_156 = (
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_157 = (
            l_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_172 = torch.nn.functional.hardtanh(
            features_171, item_156, item_157, False
        )
        features_171 = item_156 = item_157 = None
        features_173 = torch._C._nn.pad(features_172, (0, 0, 0, 0), "constant", 0.0)
        features_172 = None
        features_174 = torch.conv2d(
            features_173,
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_173 = l_self_modules_layer_modules_14_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_158 = (
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_159 = (
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_175 = torch.nn.functional.batch_norm(
            features_174,
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_158,
            item_159,
        )
        features_174 = l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_158) = (
            item_159
        ) = None
        hidden_states_9 = hidden_states_8 + features_175
        hidden_states_8 = features_175 = None
        features_176 = torch._C._nn.pad(hidden_states_9, (0, 0, 0, 0), "constant", 0.0)
        hidden_states_9 = None
        features_177 = torch.conv2d(
            features_176,
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_176 = l_self_modules_layer_modules_15_modules_expand_1x1_modules_convolution_parameters_weight_ = (None)
        item_160 = (
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_momentum = (
            None
        )
        item_161 = (
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_eps = (
            None
        )
        features_178 = torch.nn.functional.batch_norm(
            features_177,
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_bias_,
            False,
            item_160,
            item_161,
        )
        features_177 = l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_bias_ = (item_160) = (
            item_161
        ) = None
        item_162 = (
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_min_val = (
            None
        )
        item_163 = (
            l_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_max_val = (
            None
        )
        features_179 = torch.nn.functional.hardtanh(
            features_178, item_162, item_163, False
        )
        features_178 = item_162 = item_163 = None
        features_180 = torch._C._nn.pad(features_179, (1, 1, 1, 1), "constant", 0.0)
        features_179 = None
        features_181 = torch.conv2d(
            features_180,
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            240,
        )
        features_180 = l_self_modules_layer_modules_15_modules_conv_3x3_modules_convolution_parameters_weight_ = (None)
        item_164 = (
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_momentum = (
            None
        )
        item_165 = (
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_eps = (
            None
        )
        features_182 = torch.nn.functional.batch_norm(
            features_181,
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_bias_,
            False,
            item_164,
            item_165,
        )
        features_181 = l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_bias_ = (item_164) = (
            item_165
        ) = None
        item_166 = (
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_min_val.item()
        )
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_min_val = (
            None
        )
        item_167 = (
            l_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_max_val.item()
        )
        l_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_max_val = (
            None
        )
        features_183 = torch.nn.functional.hardtanh(
            features_182, item_166, item_167, False
        )
        features_182 = item_166 = item_167 = None
        features_184 = torch._C._nn.pad(features_183, (0, 0, 0, 0), "constant", 0.0)
        features_183 = None
        features_185 = torch.conv2d(
            features_184,
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_184 = l_self_modules_layer_modules_15_modules_reduce_1x1_modules_convolution_parameters_weight_ = (None)
        item_168 = (
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_momentum.item()
        )
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_momentum = (
            None
        )
        item_169 = (
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_eps.item()
        )
        l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_eps = (
            None
        )
        features_186 = torch.nn.functional.batch_norm(
            features_185,
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_bias_,
            False,
            item_168,
            item_169,
        )
        features_185 = l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_mean_ = l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_var_ = l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_weight_ = l_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_bias_ = (item_168) = (
            item_169
        ) = None
        features_187 = torch._C._nn.pad(features_186, (0, 0, 0, 0), "constant", 0.0)
        features_186 = None
        features_188 = torch.conv2d(
            features_187,
            l_self_modules_conv_1x1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_187 = (
            l_self_modules_conv_1x1_modules_convolution_parameters_weight_
        ) = None
        item_170 = l_self_modules_conv_1x1_modules_normalization_momentum.item()
        l_self_modules_conv_1x1_modules_normalization_momentum = None
        item_171 = l_self_modules_conv_1x1_modules_normalization_eps.item()
        l_self_modules_conv_1x1_modules_normalization_eps = None
        features_189 = torch.nn.functional.batch_norm(
            features_188,
            l_self_modules_conv_1x1_modules_normalization_buffers_running_mean_,
            l_self_modules_conv_1x1_modules_normalization_buffers_running_var_,
            l_self_modules_conv_1x1_modules_normalization_parameters_weight_,
            l_self_modules_conv_1x1_modules_normalization_parameters_bias_,
            False,
            item_170,
            item_171,
        )
        features_188 = (
            l_self_modules_conv_1x1_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_conv_1x1_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_conv_1x1_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_conv_1x1_modules_normalization_parameters_bias_
        ) = item_170 = item_171 = None
        item_172 = l_self_modules_conv_1x1_modules_activation_min_val.item()
        l_self_modules_conv_1x1_modules_activation_min_val = None
        item_173 = l_self_modules_conv_1x1_modules_activation_max_val.item()
        l_self_modules_conv_1x1_modules_activation_max_val = None
        features_190 = torch.nn.functional.hardtanh(
            features_189, item_172, item_173, False
        )
        features_189 = item_172 = item_173 = None
        adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(
            features_190, (1, 1)
        )
        pooled_output = torch.flatten(adaptive_avg_pool2d, start_dim=1)
        adaptive_avg_pool2d = None
        return (features_190, pooled_output)
