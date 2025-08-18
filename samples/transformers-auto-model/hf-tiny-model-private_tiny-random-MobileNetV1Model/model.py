import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s99: torch.SymInt,
        L_pixel_values_: torch.Tensor,
        L_self_modules_conv_stem_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_normalization_momentum: torch.Tensor,
        L_self_modules_conv_stem_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_stem_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_stem_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_stem_modules_normalization_eps: torch.Tensor,
        L_self_modules_conv_stem_modules_activation_min_val: torch.Tensor,
        L_self_modules_conv_stem_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_0_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_0_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_0_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_0_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_0_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_0_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_1_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_1_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_1_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_1_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_1_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_1_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_2_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_2_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_2_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_2_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_2_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_2_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_3_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_3_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_3_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_3_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_3_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_3_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_4_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_4_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_4_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_4_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_4_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_4_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_5_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_5_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_5_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_5_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_5_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_5_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_6_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_6_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_6_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_6_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_6_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_6_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_7_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_7_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_7_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_7_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_7_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_7_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_8_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_8_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_8_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_8_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_8_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_8_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_9_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_9_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_9_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_9_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_9_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_9_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_10_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_10_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_10_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_10_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_10_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_10_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_11_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_11_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_11_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_11_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_11_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_11_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_12_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_12_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_12_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_12_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_12_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_12_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_13_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_13_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_13_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_13_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_13_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_13_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_14_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_14_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_14_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_14_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_14_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_14_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_15_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_15_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_15_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_15_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_15_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_15_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_16_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_16_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_16_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_16_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_16_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_16_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_16_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_17_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_17_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_17_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_17_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_17_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_17_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_17_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_18_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_18_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_18_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_18_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_18_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_18_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_18_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_19_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_19_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_19_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_19_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_19_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_19_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_19_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_20_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_20_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_20_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_20_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_20_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_20_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_20_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_21_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_21_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_21_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_21_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_21_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_21_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_21_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_22_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_22_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_22_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_22_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_22_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_22_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_22_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_23_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_23_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_23_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_23_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_23_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_23_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_23_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_24_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_24_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_24_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_24_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_24_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_24_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_24_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_24_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_24_modules_activation_max_val: torch.Tensor,
        L_self_modules_layer_modules_25_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_25_modules_normalization_momentum: torch.Tensor,
        L_self_modules_layer_modules_25_modules_normalization_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer_modules_25_modules_normalization_buffers_running_var_: torch.Tensor,
        L_self_modules_layer_modules_25_modules_normalization_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_25_modules_normalization_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer_modules_25_modules_normalization_eps: torch.Tensor,
        L_self_modules_layer_modules_25_modules_activation_min_val: torch.Tensor,
        L_self_modules_layer_modules_25_modules_activation_max_val: torch.Tensor,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_conv_stem_modules_convolution_parameters_weight_ = (
            L_self_modules_conv_stem_modules_convolution_parameters_weight_
        )
        l_self_modules_conv_stem_modules_normalization_momentum = (
            L_self_modules_conv_stem_modules_normalization_momentum
        )
        l_self_modules_conv_stem_modules_normalization_buffers_running_mean_ = (
            L_self_modules_conv_stem_modules_normalization_buffers_running_mean_
        )
        l_self_modules_conv_stem_modules_normalization_buffers_running_var_ = (
            L_self_modules_conv_stem_modules_normalization_buffers_running_var_
        )
        l_self_modules_conv_stem_modules_normalization_parameters_weight_ = (
            L_self_modules_conv_stem_modules_normalization_parameters_weight_
        )
        l_self_modules_conv_stem_modules_normalization_parameters_bias_ = (
            L_self_modules_conv_stem_modules_normalization_parameters_bias_
        )
        l_self_modules_conv_stem_modules_normalization_eps = (
            L_self_modules_conv_stem_modules_normalization_eps
        )
        l_self_modules_conv_stem_modules_activation_min_val = (
            L_self_modules_conv_stem_modules_activation_min_val
        )
        l_self_modules_conv_stem_modules_activation_max_val = (
            L_self_modules_conv_stem_modules_activation_max_val
        )
        l_self_modules_layer_modules_0_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_0_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_0_modules_normalization_momentum = (
            L_self_modules_layer_modules_0_modules_normalization_momentum
        )
        l_self_modules_layer_modules_0_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_0_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_0_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_0_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_0_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_0_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_0_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_0_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_0_modules_normalization_eps = (
            L_self_modules_layer_modules_0_modules_normalization_eps
        )
        l_self_modules_layer_modules_0_modules_activation_min_val = (
            L_self_modules_layer_modules_0_modules_activation_min_val
        )
        l_self_modules_layer_modules_0_modules_activation_max_val = (
            L_self_modules_layer_modules_0_modules_activation_max_val
        )
        l_self_modules_layer_modules_1_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_1_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_1_modules_normalization_momentum = (
            L_self_modules_layer_modules_1_modules_normalization_momentum
        )
        l_self_modules_layer_modules_1_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_1_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_1_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_1_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_1_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_1_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_1_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_1_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_1_modules_normalization_eps = (
            L_self_modules_layer_modules_1_modules_normalization_eps
        )
        l_self_modules_layer_modules_1_modules_activation_min_val = (
            L_self_modules_layer_modules_1_modules_activation_min_val
        )
        l_self_modules_layer_modules_1_modules_activation_max_val = (
            L_self_modules_layer_modules_1_modules_activation_max_val
        )
        l_self_modules_layer_modules_2_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_2_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_2_modules_normalization_momentum = (
            L_self_modules_layer_modules_2_modules_normalization_momentum
        )
        l_self_modules_layer_modules_2_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_2_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_2_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_2_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_2_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_2_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_2_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_2_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_2_modules_normalization_eps = (
            L_self_modules_layer_modules_2_modules_normalization_eps
        )
        l_self_modules_layer_modules_2_modules_activation_min_val = (
            L_self_modules_layer_modules_2_modules_activation_min_val
        )
        l_self_modules_layer_modules_2_modules_activation_max_val = (
            L_self_modules_layer_modules_2_modules_activation_max_val
        )
        l_self_modules_layer_modules_3_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_3_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_3_modules_normalization_momentum = (
            L_self_modules_layer_modules_3_modules_normalization_momentum
        )
        l_self_modules_layer_modules_3_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_3_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_3_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_3_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_3_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_3_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_3_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_3_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_3_modules_normalization_eps = (
            L_self_modules_layer_modules_3_modules_normalization_eps
        )
        l_self_modules_layer_modules_3_modules_activation_min_val = (
            L_self_modules_layer_modules_3_modules_activation_min_val
        )
        l_self_modules_layer_modules_3_modules_activation_max_val = (
            L_self_modules_layer_modules_3_modules_activation_max_val
        )
        l_self_modules_layer_modules_4_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_4_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_4_modules_normalization_momentum = (
            L_self_modules_layer_modules_4_modules_normalization_momentum
        )
        l_self_modules_layer_modules_4_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_4_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_4_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_4_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_4_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_4_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_4_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_4_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_4_modules_normalization_eps = (
            L_self_modules_layer_modules_4_modules_normalization_eps
        )
        l_self_modules_layer_modules_4_modules_activation_min_val = (
            L_self_modules_layer_modules_4_modules_activation_min_val
        )
        l_self_modules_layer_modules_4_modules_activation_max_val = (
            L_self_modules_layer_modules_4_modules_activation_max_val
        )
        l_self_modules_layer_modules_5_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_5_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_5_modules_normalization_momentum = (
            L_self_modules_layer_modules_5_modules_normalization_momentum
        )
        l_self_modules_layer_modules_5_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_5_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_5_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_5_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_5_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_5_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_5_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_5_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_5_modules_normalization_eps = (
            L_self_modules_layer_modules_5_modules_normalization_eps
        )
        l_self_modules_layer_modules_5_modules_activation_min_val = (
            L_self_modules_layer_modules_5_modules_activation_min_val
        )
        l_self_modules_layer_modules_5_modules_activation_max_val = (
            L_self_modules_layer_modules_5_modules_activation_max_val
        )
        l_self_modules_layer_modules_6_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_6_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_6_modules_normalization_momentum = (
            L_self_modules_layer_modules_6_modules_normalization_momentum
        )
        l_self_modules_layer_modules_6_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_6_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_6_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_6_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_6_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_6_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_6_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_6_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_6_modules_normalization_eps = (
            L_self_modules_layer_modules_6_modules_normalization_eps
        )
        l_self_modules_layer_modules_6_modules_activation_min_val = (
            L_self_modules_layer_modules_6_modules_activation_min_val
        )
        l_self_modules_layer_modules_6_modules_activation_max_val = (
            L_self_modules_layer_modules_6_modules_activation_max_val
        )
        l_self_modules_layer_modules_7_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_7_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_7_modules_normalization_momentum = (
            L_self_modules_layer_modules_7_modules_normalization_momentum
        )
        l_self_modules_layer_modules_7_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_7_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_7_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_7_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_7_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_7_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_7_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_7_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_7_modules_normalization_eps = (
            L_self_modules_layer_modules_7_modules_normalization_eps
        )
        l_self_modules_layer_modules_7_modules_activation_min_val = (
            L_self_modules_layer_modules_7_modules_activation_min_val
        )
        l_self_modules_layer_modules_7_modules_activation_max_val = (
            L_self_modules_layer_modules_7_modules_activation_max_val
        )
        l_self_modules_layer_modules_8_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_8_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_8_modules_normalization_momentum = (
            L_self_modules_layer_modules_8_modules_normalization_momentum
        )
        l_self_modules_layer_modules_8_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_8_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_8_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_8_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_8_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_8_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_8_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_8_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_8_modules_normalization_eps = (
            L_self_modules_layer_modules_8_modules_normalization_eps
        )
        l_self_modules_layer_modules_8_modules_activation_min_val = (
            L_self_modules_layer_modules_8_modules_activation_min_val
        )
        l_self_modules_layer_modules_8_modules_activation_max_val = (
            L_self_modules_layer_modules_8_modules_activation_max_val
        )
        l_self_modules_layer_modules_9_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_9_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_9_modules_normalization_momentum = (
            L_self_modules_layer_modules_9_modules_normalization_momentum
        )
        l_self_modules_layer_modules_9_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_9_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_9_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_9_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_9_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_9_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_9_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_9_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_9_modules_normalization_eps = (
            L_self_modules_layer_modules_9_modules_normalization_eps
        )
        l_self_modules_layer_modules_9_modules_activation_min_val = (
            L_self_modules_layer_modules_9_modules_activation_min_val
        )
        l_self_modules_layer_modules_9_modules_activation_max_val = (
            L_self_modules_layer_modules_9_modules_activation_max_val
        )
        l_self_modules_layer_modules_10_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_10_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_10_modules_normalization_momentum = (
            L_self_modules_layer_modules_10_modules_normalization_momentum
        )
        l_self_modules_layer_modules_10_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_10_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_10_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_10_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_10_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_10_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_10_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_10_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_10_modules_normalization_eps = (
            L_self_modules_layer_modules_10_modules_normalization_eps
        )
        l_self_modules_layer_modules_10_modules_activation_min_val = (
            L_self_modules_layer_modules_10_modules_activation_min_val
        )
        l_self_modules_layer_modules_10_modules_activation_max_val = (
            L_self_modules_layer_modules_10_modules_activation_max_val
        )
        l_self_modules_layer_modules_11_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_11_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_11_modules_normalization_momentum = (
            L_self_modules_layer_modules_11_modules_normalization_momentum
        )
        l_self_modules_layer_modules_11_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_11_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_11_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_11_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_11_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_11_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_11_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_11_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_11_modules_normalization_eps = (
            L_self_modules_layer_modules_11_modules_normalization_eps
        )
        l_self_modules_layer_modules_11_modules_activation_min_val = (
            L_self_modules_layer_modules_11_modules_activation_min_val
        )
        l_self_modules_layer_modules_11_modules_activation_max_val = (
            L_self_modules_layer_modules_11_modules_activation_max_val
        )
        l_self_modules_layer_modules_12_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_12_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_12_modules_normalization_momentum = (
            L_self_modules_layer_modules_12_modules_normalization_momentum
        )
        l_self_modules_layer_modules_12_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_12_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_12_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_12_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_12_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_12_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_12_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_12_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_12_modules_normalization_eps = (
            L_self_modules_layer_modules_12_modules_normalization_eps
        )
        l_self_modules_layer_modules_12_modules_activation_min_val = (
            L_self_modules_layer_modules_12_modules_activation_min_val
        )
        l_self_modules_layer_modules_12_modules_activation_max_val = (
            L_self_modules_layer_modules_12_modules_activation_max_val
        )
        l_self_modules_layer_modules_13_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_13_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_13_modules_normalization_momentum = (
            L_self_modules_layer_modules_13_modules_normalization_momentum
        )
        l_self_modules_layer_modules_13_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_13_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_13_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_13_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_13_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_13_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_13_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_13_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_13_modules_normalization_eps = (
            L_self_modules_layer_modules_13_modules_normalization_eps
        )
        l_self_modules_layer_modules_13_modules_activation_min_val = (
            L_self_modules_layer_modules_13_modules_activation_min_val
        )
        l_self_modules_layer_modules_13_modules_activation_max_val = (
            L_self_modules_layer_modules_13_modules_activation_max_val
        )
        l_self_modules_layer_modules_14_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_14_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_14_modules_normalization_momentum = (
            L_self_modules_layer_modules_14_modules_normalization_momentum
        )
        l_self_modules_layer_modules_14_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_14_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_14_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_14_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_14_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_14_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_14_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_14_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_14_modules_normalization_eps = (
            L_self_modules_layer_modules_14_modules_normalization_eps
        )
        l_self_modules_layer_modules_14_modules_activation_min_val = (
            L_self_modules_layer_modules_14_modules_activation_min_val
        )
        l_self_modules_layer_modules_14_modules_activation_max_val = (
            L_self_modules_layer_modules_14_modules_activation_max_val
        )
        l_self_modules_layer_modules_15_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_15_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_15_modules_normalization_momentum = (
            L_self_modules_layer_modules_15_modules_normalization_momentum
        )
        l_self_modules_layer_modules_15_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_15_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_15_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_15_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_15_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_15_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_15_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_15_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_15_modules_normalization_eps = (
            L_self_modules_layer_modules_15_modules_normalization_eps
        )
        l_self_modules_layer_modules_15_modules_activation_min_val = (
            L_self_modules_layer_modules_15_modules_activation_min_val
        )
        l_self_modules_layer_modules_15_modules_activation_max_val = (
            L_self_modules_layer_modules_15_modules_activation_max_val
        )
        l_self_modules_layer_modules_16_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_16_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_16_modules_normalization_momentum = (
            L_self_modules_layer_modules_16_modules_normalization_momentum
        )
        l_self_modules_layer_modules_16_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_16_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_16_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_16_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_16_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_16_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_16_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_16_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_16_modules_normalization_eps = (
            L_self_modules_layer_modules_16_modules_normalization_eps
        )
        l_self_modules_layer_modules_16_modules_activation_min_val = (
            L_self_modules_layer_modules_16_modules_activation_min_val
        )
        l_self_modules_layer_modules_16_modules_activation_max_val = (
            L_self_modules_layer_modules_16_modules_activation_max_val
        )
        l_self_modules_layer_modules_17_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_17_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_17_modules_normalization_momentum = (
            L_self_modules_layer_modules_17_modules_normalization_momentum
        )
        l_self_modules_layer_modules_17_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_17_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_17_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_17_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_17_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_17_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_17_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_17_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_17_modules_normalization_eps = (
            L_self_modules_layer_modules_17_modules_normalization_eps
        )
        l_self_modules_layer_modules_17_modules_activation_min_val = (
            L_self_modules_layer_modules_17_modules_activation_min_val
        )
        l_self_modules_layer_modules_17_modules_activation_max_val = (
            L_self_modules_layer_modules_17_modules_activation_max_val
        )
        l_self_modules_layer_modules_18_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_18_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_18_modules_normalization_momentum = (
            L_self_modules_layer_modules_18_modules_normalization_momentum
        )
        l_self_modules_layer_modules_18_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_18_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_18_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_18_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_18_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_18_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_18_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_18_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_18_modules_normalization_eps = (
            L_self_modules_layer_modules_18_modules_normalization_eps
        )
        l_self_modules_layer_modules_18_modules_activation_min_val = (
            L_self_modules_layer_modules_18_modules_activation_min_val
        )
        l_self_modules_layer_modules_18_modules_activation_max_val = (
            L_self_modules_layer_modules_18_modules_activation_max_val
        )
        l_self_modules_layer_modules_19_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_19_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_19_modules_normalization_momentum = (
            L_self_modules_layer_modules_19_modules_normalization_momentum
        )
        l_self_modules_layer_modules_19_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_19_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_19_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_19_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_19_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_19_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_19_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_19_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_19_modules_normalization_eps = (
            L_self_modules_layer_modules_19_modules_normalization_eps
        )
        l_self_modules_layer_modules_19_modules_activation_min_val = (
            L_self_modules_layer_modules_19_modules_activation_min_val
        )
        l_self_modules_layer_modules_19_modules_activation_max_val = (
            L_self_modules_layer_modules_19_modules_activation_max_val
        )
        l_self_modules_layer_modules_20_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_20_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_20_modules_normalization_momentum = (
            L_self_modules_layer_modules_20_modules_normalization_momentum
        )
        l_self_modules_layer_modules_20_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_20_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_20_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_20_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_20_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_20_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_20_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_20_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_20_modules_normalization_eps = (
            L_self_modules_layer_modules_20_modules_normalization_eps
        )
        l_self_modules_layer_modules_20_modules_activation_min_val = (
            L_self_modules_layer_modules_20_modules_activation_min_val
        )
        l_self_modules_layer_modules_20_modules_activation_max_val = (
            L_self_modules_layer_modules_20_modules_activation_max_val
        )
        l_self_modules_layer_modules_21_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_21_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_21_modules_normalization_momentum = (
            L_self_modules_layer_modules_21_modules_normalization_momentum
        )
        l_self_modules_layer_modules_21_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_21_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_21_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_21_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_21_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_21_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_21_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_21_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_21_modules_normalization_eps = (
            L_self_modules_layer_modules_21_modules_normalization_eps
        )
        l_self_modules_layer_modules_21_modules_activation_min_val = (
            L_self_modules_layer_modules_21_modules_activation_min_val
        )
        l_self_modules_layer_modules_21_modules_activation_max_val = (
            L_self_modules_layer_modules_21_modules_activation_max_val
        )
        l_self_modules_layer_modules_22_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_22_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_22_modules_normalization_momentum = (
            L_self_modules_layer_modules_22_modules_normalization_momentum
        )
        l_self_modules_layer_modules_22_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_22_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_22_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_22_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_22_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_22_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_22_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_22_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_22_modules_normalization_eps = (
            L_self_modules_layer_modules_22_modules_normalization_eps
        )
        l_self_modules_layer_modules_22_modules_activation_min_val = (
            L_self_modules_layer_modules_22_modules_activation_min_val
        )
        l_self_modules_layer_modules_22_modules_activation_max_val = (
            L_self_modules_layer_modules_22_modules_activation_max_val
        )
        l_self_modules_layer_modules_23_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_23_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_23_modules_normalization_momentum = (
            L_self_modules_layer_modules_23_modules_normalization_momentum
        )
        l_self_modules_layer_modules_23_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_23_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_23_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_23_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_23_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_23_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_23_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_23_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_23_modules_normalization_eps = (
            L_self_modules_layer_modules_23_modules_normalization_eps
        )
        l_self_modules_layer_modules_23_modules_activation_min_val = (
            L_self_modules_layer_modules_23_modules_activation_min_val
        )
        l_self_modules_layer_modules_23_modules_activation_max_val = (
            L_self_modules_layer_modules_23_modules_activation_max_val
        )
        l_self_modules_layer_modules_24_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_24_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_24_modules_normalization_momentum = (
            L_self_modules_layer_modules_24_modules_normalization_momentum
        )
        l_self_modules_layer_modules_24_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_24_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_24_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_24_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_24_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_24_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_24_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_24_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_24_modules_normalization_eps = (
            L_self_modules_layer_modules_24_modules_normalization_eps
        )
        l_self_modules_layer_modules_24_modules_activation_min_val = (
            L_self_modules_layer_modules_24_modules_activation_min_val
        )
        l_self_modules_layer_modules_24_modules_activation_max_val = (
            L_self_modules_layer_modules_24_modules_activation_max_val
        )
        l_self_modules_layer_modules_25_modules_convolution_parameters_weight_ = (
            L_self_modules_layer_modules_25_modules_convolution_parameters_weight_
        )
        l_self_modules_layer_modules_25_modules_normalization_momentum = (
            L_self_modules_layer_modules_25_modules_normalization_momentum
        )
        l_self_modules_layer_modules_25_modules_normalization_buffers_running_mean_ = (
            L_self_modules_layer_modules_25_modules_normalization_buffers_running_mean_
        )
        l_self_modules_layer_modules_25_modules_normalization_buffers_running_var_ = (
            L_self_modules_layer_modules_25_modules_normalization_buffers_running_var_
        )
        l_self_modules_layer_modules_25_modules_normalization_parameters_weight_ = (
            L_self_modules_layer_modules_25_modules_normalization_parameters_weight_
        )
        l_self_modules_layer_modules_25_modules_normalization_parameters_bias_ = (
            L_self_modules_layer_modules_25_modules_normalization_parameters_bias_
        )
        l_self_modules_layer_modules_25_modules_normalization_eps = (
            L_self_modules_layer_modules_25_modules_normalization_eps
        )
        l_self_modules_layer_modules_25_modules_activation_min_val = (
            L_self_modules_layer_modules_25_modules_activation_min_val
        )
        l_self_modules_layer_modules_25_modules_activation_max_val = (
            L_self_modules_layer_modules_25_modules_activation_max_val
        )
        features = torch._C._nn.pad(l_pixel_values_, (0, 1, 0, 1), "constant", 0.0)
        l_pixel_values_ = None
        features_1 = torch.conv2d(
            features,
            l_self_modules_conv_stem_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        features = (
            l_self_modules_conv_stem_modules_convolution_parameters_weight_
        ) = None
        item = l_self_modules_conv_stem_modules_normalization_momentum.item()
        l_self_modules_conv_stem_modules_normalization_momentum = None
        item_1 = l_self_modules_conv_stem_modules_normalization_eps.item()
        l_self_modules_conv_stem_modules_normalization_eps = None
        features_2 = torch.nn.functional.batch_norm(
            features_1,
            l_self_modules_conv_stem_modules_normalization_buffers_running_mean_,
            l_self_modules_conv_stem_modules_normalization_buffers_running_var_,
            l_self_modules_conv_stem_modules_normalization_parameters_weight_,
            l_self_modules_conv_stem_modules_normalization_parameters_bias_,
            False,
            item,
            item_1,
        )
        features_1 = (
            l_self_modules_conv_stem_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_conv_stem_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_conv_stem_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_conv_stem_modules_normalization_parameters_bias_
        ) = item = item_1 = None
        item_2 = l_self_modules_conv_stem_modules_activation_min_val.item()
        l_self_modules_conv_stem_modules_activation_min_val = None
        item_3 = l_self_modules_conv_stem_modules_activation_max_val.item()
        l_self_modules_conv_stem_modules_activation_max_val = None
        features_3 = torch.nn.functional.hardtanh(features_2, item_2, item_3, False)
        features_2 = item_2 = item_3 = None
        features_4 = torch._C._nn.pad(features_3, (1, 1, 1, 1), "constant", 0.0)
        features_3 = None
        features_5 = torch.conv2d(
            features_4,
            l_self_modules_layer_modules_0_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            8,
        )
        features_4 = (
            l_self_modules_layer_modules_0_modules_convolution_parameters_weight_
        ) = None
        item_4 = l_self_modules_layer_modules_0_modules_normalization_momentum.item()
        l_self_modules_layer_modules_0_modules_normalization_momentum = None
        item_5 = l_self_modules_layer_modules_0_modules_normalization_eps.item()
        l_self_modules_layer_modules_0_modules_normalization_eps = None
        features_6 = torch.nn.functional.batch_norm(
            features_5,
            l_self_modules_layer_modules_0_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_0_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_0_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_0_modules_normalization_parameters_bias_,
            False,
            item_4,
            item_5,
        )
        features_5 = (
            l_self_modules_layer_modules_0_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_0_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_0_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_0_modules_normalization_parameters_bias_
        ) = item_4 = item_5 = None
        item_6 = l_self_modules_layer_modules_0_modules_activation_min_val.item()
        l_self_modules_layer_modules_0_modules_activation_min_val = None
        item_7 = l_self_modules_layer_modules_0_modules_activation_max_val.item()
        l_self_modules_layer_modules_0_modules_activation_max_val = None
        features_7 = torch.nn.functional.hardtanh(features_6, item_6, item_7, False)
        features_6 = item_6 = item_7 = None
        features_8 = torch._C._nn.pad(features_7, (0, 0, 0, 0), "constant", 0.0)
        features_7 = None
        features_9 = torch.conv2d(
            features_8,
            l_self_modules_layer_modules_1_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_8 = (
            l_self_modules_layer_modules_1_modules_convolution_parameters_weight_
        ) = None
        item_8 = l_self_modules_layer_modules_1_modules_normalization_momentum.item()
        l_self_modules_layer_modules_1_modules_normalization_momentum = None
        item_9 = l_self_modules_layer_modules_1_modules_normalization_eps.item()
        l_self_modules_layer_modules_1_modules_normalization_eps = None
        features_10 = torch.nn.functional.batch_norm(
            features_9,
            l_self_modules_layer_modules_1_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_1_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_1_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_1_modules_normalization_parameters_bias_,
            False,
            item_8,
            item_9,
        )
        features_9 = (
            l_self_modules_layer_modules_1_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_1_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_1_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_1_modules_normalization_parameters_bias_
        ) = item_8 = item_9 = None
        item_10 = l_self_modules_layer_modules_1_modules_activation_min_val.item()
        l_self_modules_layer_modules_1_modules_activation_min_val = None
        item_11 = l_self_modules_layer_modules_1_modules_activation_max_val.item()
        l_self_modules_layer_modules_1_modules_activation_max_val = None
        features_11 = torch.nn.functional.hardtanh(features_10, item_10, item_11, False)
        features_10 = item_10 = item_11 = None
        features_12 = torch._C._nn.pad(features_11, (0, 1, 0, 1), "constant", 0.0)
        features_11 = None
        features_13 = torch.conv2d(
            features_12,
            l_self_modules_layer_modules_2_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            16,
        )
        features_12 = (
            l_self_modules_layer_modules_2_modules_convolution_parameters_weight_
        ) = None
        item_12 = l_self_modules_layer_modules_2_modules_normalization_momentum.item()
        l_self_modules_layer_modules_2_modules_normalization_momentum = None
        item_13 = l_self_modules_layer_modules_2_modules_normalization_eps.item()
        l_self_modules_layer_modules_2_modules_normalization_eps = None
        features_14 = torch.nn.functional.batch_norm(
            features_13,
            l_self_modules_layer_modules_2_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_2_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_2_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_2_modules_normalization_parameters_bias_,
            False,
            item_12,
            item_13,
        )
        features_13 = (
            l_self_modules_layer_modules_2_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_2_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_2_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_2_modules_normalization_parameters_bias_
        ) = item_12 = item_13 = None
        item_14 = l_self_modules_layer_modules_2_modules_activation_min_val.item()
        l_self_modules_layer_modules_2_modules_activation_min_val = None
        item_15 = l_self_modules_layer_modules_2_modules_activation_max_val.item()
        l_self_modules_layer_modules_2_modules_activation_max_val = None
        features_15 = torch.nn.functional.hardtanh(features_14, item_14, item_15, False)
        features_14 = item_14 = item_15 = None
        features_16 = torch._C._nn.pad(features_15, (0, 0, 0, 0), "constant", 0.0)
        features_15 = None
        features_17 = torch.conv2d(
            features_16,
            l_self_modules_layer_modules_3_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_16 = (
            l_self_modules_layer_modules_3_modules_convolution_parameters_weight_
        ) = None
        item_16 = l_self_modules_layer_modules_3_modules_normalization_momentum.item()
        l_self_modules_layer_modules_3_modules_normalization_momentum = None
        item_17 = l_self_modules_layer_modules_3_modules_normalization_eps.item()
        l_self_modules_layer_modules_3_modules_normalization_eps = None
        features_18 = torch.nn.functional.batch_norm(
            features_17,
            l_self_modules_layer_modules_3_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_3_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_3_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_3_modules_normalization_parameters_bias_,
            False,
            item_16,
            item_17,
        )
        features_17 = (
            l_self_modules_layer_modules_3_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_3_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_3_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_3_modules_normalization_parameters_bias_
        ) = item_16 = item_17 = None
        item_18 = l_self_modules_layer_modules_3_modules_activation_min_val.item()
        l_self_modules_layer_modules_3_modules_activation_min_val = None
        item_19 = l_self_modules_layer_modules_3_modules_activation_max_val.item()
        l_self_modules_layer_modules_3_modules_activation_max_val = None
        features_19 = torch.nn.functional.hardtanh(features_18, item_18, item_19, False)
        features_18 = item_18 = item_19 = None
        features_20 = torch._C._nn.pad(features_19, (1, 1, 1, 1), "constant", 0.0)
        features_19 = None
        features_21 = torch.conv2d(
            features_20,
            l_self_modules_layer_modules_4_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            32,
        )
        features_20 = (
            l_self_modules_layer_modules_4_modules_convolution_parameters_weight_
        ) = None
        item_20 = l_self_modules_layer_modules_4_modules_normalization_momentum.item()
        l_self_modules_layer_modules_4_modules_normalization_momentum = None
        item_21 = l_self_modules_layer_modules_4_modules_normalization_eps.item()
        l_self_modules_layer_modules_4_modules_normalization_eps = None
        features_22 = torch.nn.functional.batch_norm(
            features_21,
            l_self_modules_layer_modules_4_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_4_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_4_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_4_modules_normalization_parameters_bias_,
            False,
            item_20,
            item_21,
        )
        features_21 = (
            l_self_modules_layer_modules_4_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_4_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_4_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_4_modules_normalization_parameters_bias_
        ) = item_20 = item_21 = None
        item_22 = l_self_modules_layer_modules_4_modules_activation_min_val.item()
        l_self_modules_layer_modules_4_modules_activation_min_val = None
        item_23 = l_self_modules_layer_modules_4_modules_activation_max_val.item()
        l_self_modules_layer_modules_4_modules_activation_max_val = None
        features_23 = torch.nn.functional.hardtanh(features_22, item_22, item_23, False)
        features_22 = item_22 = item_23 = None
        features_24 = torch._C._nn.pad(features_23, (0, 0, 0, 0), "constant", 0.0)
        features_23 = None
        features_25 = torch.conv2d(
            features_24,
            l_self_modules_layer_modules_5_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_24 = (
            l_self_modules_layer_modules_5_modules_convolution_parameters_weight_
        ) = None
        item_24 = l_self_modules_layer_modules_5_modules_normalization_momentum.item()
        l_self_modules_layer_modules_5_modules_normalization_momentum = None
        item_25 = l_self_modules_layer_modules_5_modules_normalization_eps.item()
        l_self_modules_layer_modules_5_modules_normalization_eps = None
        features_26 = torch.nn.functional.batch_norm(
            features_25,
            l_self_modules_layer_modules_5_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_5_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_5_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_5_modules_normalization_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        features_25 = (
            l_self_modules_layer_modules_5_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_5_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_5_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_5_modules_normalization_parameters_bias_
        ) = item_24 = item_25 = None
        item_26 = l_self_modules_layer_modules_5_modules_activation_min_val.item()
        l_self_modules_layer_modules_5_modules_activation_min_val = None
        item_27 = l_self_modules_layer_modules_5_modules_activation_max_val.item()
        l_self_modules_layer_modules_5_modules_activation_max_val = None
        features_27 = torch.nn.functional.hardtanh(features_26, item_26, item_27, False)
        features_26 = item_26 = item_27 = None
        features_28 = torch._C._nn.pad(features_27, (0, 1, 0, 1), "constant", 0.0)
        features_27 = None
        features_29 = torch.conv2d(
            features_28,
            l_self_modules_layer_modules_6_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            32,
        )
        features_28 = (
            l_self_modules_layer_modules_6_modules_convolution_parameters_weight_
        ) = None
        item_28 = l_self_modules_layer_modules_6_modules_normalization_momentum.item()
        l_self_modules_layer_modules_6_modules_normalization_momentum = None
        item_29 = l_self_modules_layer_modules_6_modules_normalization_eps.item()
        l_self_modules_layer_modules_6_modules_normalization_eps = None
        features_30 = torch.nn.functional.batch_norm(
            features_29,
            l_self_modules_layer_modules_6_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_6_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_6_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_6_modules_normalization_parameters_bias_,
            False,
            item_28,
            item_29,
        )
        features_29 = (
            l_self_modules_layer_modules_6_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_6_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_6_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_6_modules_normalization_parameters_bias_
        ) = item_28 = item_29 = None
        item_30 = l_self_modules_layer_modules_6_modules_activation_min_val.item()
        l_self_modules_layer_modules_6_modules_activation_min_val = None
        item_31 = l_self_modules_layer_modules_6_modules_activation_max_val.item()
        l_self_modules_layer_modules_6_modules_activation_max_val = None
        features_31 = torch.nn.functional.hardtanh(features_30, item_30, item_31, False)
        features_30 = item_30 = item_31 = None
        features_32 = torch._C._nn.pad(features_31, (0, 0, 0, 0), "constant", 0.0)
        features_31 = None
        features_33 = torch.conv2d(
            features_32,
            l_self_modules_layer_modules_7_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_32 = (
            l_self_modules_layer_modules_7_modules_convolution_parameters_weight_
        ) = None
        item_32 = l_self_modules_layer_modules_7_modules_normalization_momentum.item()
        l_self_modules_layer_modules_7_modules_normalization_momentum = None
        item_33 = l_self_modules_layer_modules_7_modules_normalization_eps.item()
        l_self_modules_layer_modules_7_modules_normalization_eps = None
        features_34 = torch.nn.functional.batch_norm(
            features_33,
            l_self_modules_layer_modules_7_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_7_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_7_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_7_modules_normalization_parameters_bias_,
            False,
            item_32,
            item_33,
        )
        features_33 = (
            l_self_modules_layer_modules_7_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_7_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_7_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_7_modules_normalization_parameters_bias_
        ) = item_32 = item_33 = None
        item_34 = l_self_modules_layer_modules_7_modules_activation_min_val.item()
        l_self_modules_layer_modules_7_modules_activation_min_val = None
        item_35 = l_self_modules_layer_modules_7_modules_activation_max_val.item()
        l_self_modules_layer_modules_7_modules_activation_max_val = None
        features_35 = torch.nn.functional.hardtanh(features_34, item_34, item_35, False)
        features_34 = item_34 = item_35 = None
        features_36 = torch._C._nn.pad(features_35, (1, 1, 1, 1), "constant", 0.0)
        features_35 = None
        features_37 = torch.conv2d(
            features_36,
            l_self_modules_layer_modules_8_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            64,
        )
        features_36 = (
            l_self_modules_layer_modules_8_modules_convolution_parameters_weight_
        ) = None
        item_36 = l_self_modules_layer_modules_8_modules_normalization_momentum.item()
        l_self_modules_layer_modules_8_modules_normalization_momentum = None
        item_37 = l_self_modules_layer_modules_8_modules_normalization_eps.item()
        l_self_modules_layer_modules_8_modules_normalization_eps = None
        features_38 = torch.nn.functional.batch_norm(
            features_37,
            l_self_modules_layer_modules_8_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_8_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_8_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_8_modules_normalization_parameters_bias_,
            False,
            item_36,
            item_37,
        )
        features_37 = (
            l_self_modules_layer_modules_8_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_8_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_8_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_8_modules_normalization_parameters_bias_
        ) = item_36 = item_37 = None
        item_38 = l_self_modules_layer_modules_8_modules_activation_min_val.item()
        l_self_modules_layer_modules_8_modules_activation_min_val = None
        item_39 = l_self_modules_layer_modules_8_modules_activation_max_val.item()
        l_self_modules_layer_modules_8_modules_activation_max_val = None
        features_39 = torch.nn.functional.hardtanh(features_38, item_38, item_39, False)
        features_38 = item_38 = item_39 = None
        features_40 = torch._C._nn.pad(features_39, (0, 0, 0, 0), "constant", 0.0)
        features_39 = None
        features_41 = torch.conv2d(
            features_40,
            l_self_modules_layer_modules_9_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_40 = (
            l_self_modules_layer_modules_9_modules_convolution_parameters_weight_
        ) = None
        item_40 = l_self_modules_layer_modules_9_modules_normalization_momentum.item()
        l_self_modules_layer_modules_9_modules_normalization_momentum = None
        item_41 = l_self_modules_layer_modules_9_modules_normalization_eps.item()
        l_self_modules_layer_modules_9_modules_normalization_eps = None
        features_42 = torch.nn.functional.batch_norm(
            features_41,
            l_self_modules_layer_modules_9_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_9_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_9_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_9_modules_normalization_parameters_bias_,
            False,
            item_40,
            item_41,
        )
        features_41 = (
            l_self_modules_layer_modules_9_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_9_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_9_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_9_modules_normalization_parameters_bias_
        ) = item_40 = item_41 = None
        item_42 = l_self_modules_layer_modules_9_modules_activation_min_val.item()
        l_self_modules_layer_modules_9_modules_activation_min_val = None
        item_43 = l_self_modules_layer_modules_9_modules_activation_max_val.item()
        l_self_modules_layer_modules_9_modules_activation_max_val = None
        features_43 = torch.nn.functional.hardtanh(features_42, item_42, item_43, False)
        features_42 = item_42 = item_43 = None
        features_44 = torch._C._nn.pad(features_43, (0, 1, 0, 1), "constant", 0.0)
        features_43 = None
        features_45 = torch.conv2d(
            features_44,
            l_self_modules_layer_modules_10_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            64,
        )
        features_44 = (
            l_self_modules_layer_modules_10_modules_convolution_parameters_weight_
        ) = None
        item_44 = l_self_modules_layer_modules_10_modules_normalization_momentum.item()
        l_self_modules_layer_modules_10_modules_normalization_momentum = None
        item_45 = l_self_modules_layer_modules_10_modules_normalization_eps.item()
        l_self_modules_layer_modules_10_modules_normalization_eps = None
        features_46 = torch.nn.functional.batch_norm(
            features_45,
            l_self_modules_layer_modules_10_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_10_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_10_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_10_modules_normalization_parameters_bias_,
            False,
            item_44,
            item_45,
        )
        features_45 = (
            l_self_modules_layer_modules_10_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_10_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_10_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_10_modules_normalization_parameters_bias_
        ) = item_44 = item_45 = None
        item_46 = l_self_modules_layer_modules_10_modules_activation_min_val.item()
        l_self_modules_layer_modules_10_modules_activation_min_val = None
        item_47 = l_self_modules_layer_modules_10_modules_activation_max_val.item()
        l_self_modules_layer_modules_10_modules_activation_max_val = None
        features_47 = torch.nn.functional.hardtanh(features_46, item_46, item_47, False)
        features_46 = item_46 = item_47 = None
        features_48 = torch._C._nn.pad(features_47, (0, 0, 0, 0), "constant", 0.0)
        features_47 = None
        features_49 = torch.conv2d(
            features_48,
            l_self_modules_layer_modules_11_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_48 = (
            l_self_modules_layer_modules_11_modules_convolution_parameters_weight_
        ) = None
        item_48 = l_self_modules_layer_modules_11_modules_normalization_momentum.item()
        l_self_modules_layer_modules_11_modules_normalization_momentum = None
        item_49 = l_self_modules_layer_modules_11_modules_normalization_eps.item()
        l_self_modules_layer_modules_11_modules_normalization_eps = None
        features_50 = torch.nn.functional.batch_norm(
            features_49,
            l_self_modules_layer_modules_11_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_11_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_11_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_11_modules_normalization_parameters_bias_,
            False,
            item_48,
            item_49,
        )
        features_49 = (
            l_self_modules_layer_modules_11_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_11_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_11_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_11_modules_normalization_parameters_bias_
        ) = item_48 = item_49 = None
        item_50 = l_self_modules_layer_modules_11_modules_activation_min_val.item()
        l_self_modules_layer_modules_11_modules_activation_min_val = None
        item_51 = l_self_modules_layer_modules_11_modules_activation_max_val.item()
        l_self_modules_layer_modules_11_modules_activation_max_val = None
        features_51 = torch.nn.functional.hardtanh(features_50, item_50, item_51, False)
        features_50 = item_50 = item_51 = None
        features_52 = torch._C._nn.pad(features_51, (1, 1, 1, 1), "constant", 0.0)
        features_51 = None
        features_53 = torch.conv2d(
            features_52,
            l_self_modules_layer_modules_12_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        features_52 = (
            l_self_modules_layer_modules_12_modules_convolution_parameters_weight_
        ) = None
        item_52 = l_self_modules_layer_modules_12_modules_normalization_momentum.item()
        l_self_modules_layer_modules_12_modules_normalization_momentum = None
        item_53 = l_self_modules_layer_modules_12_modules_normalization_eps.item()
        l_self_modules_layer_modules_12_modules_normalization_eps = None
        features_54 = torch.nn.functional.batch_norm(
            features_53,
            l_self_modules_layer_modules_12_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_12_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_12_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_12_modules_normalization_parameters_bias_,
            False,
            item_52,
            item_53,
        )
        features_53 = (
            l_self_modules_layer_modules_12_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_12_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_12_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_12_modules_normalization_parameters_bias_
        ) = item_52 = item_53 = None
        item_54 = l_self_modules_layer_modules_12_modules_activation_min_val.item()
        l_self_modules_layer_modules_12_modules_activation_min_val = None
        item_55 = l_self_modules_layer_modules_12_modules_activation_max_val.item()
        l_self_modules_layer_modules_12_modules_activation_max_val = None
        features_55 = torch.nn.functional.hardtanh(features_54, item_54, item_55, False)
        features_54 = item_54 = item_55 = None
        features_56 = torch._C._nn.pad(features_55, (0, 0, 0, 0), "constant", 0.0)
        features_55 = None
        features_57 = torch.conv2d(
            features_56,
            l_self_modules_layer_modules_13_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_56 = (
            l_self_modules_layer_modules_13_modules_convolution_parameters_weight_
        ) = None
        item_56 = l_self_modules_layer_modules_13_modules_normalization_momentum.item()
        l_self_modules_layer_modules_13_modules_normalization_momentum = None
        item_57 = l_self_modules_layer_modules_13_modules_normalization_eps.item()
        l_self_modules_layer_modules_13_modules_normalization_eps = None
        features_58 = torch.nn.functional.batch_norm(
            features_57,
            l_self_modules_layer_modules_13_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_13_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_13_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_13_modules_normalization_parameters_bias_,
            False,
            item_56,
            item_57,
        )
        features_57 = (
            l_self_modules_layer_modules_13_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_13_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_13_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_13_modules_normalization_parameters_bias_
        ) = item_56 = item_57 = None
        item_58 = l_self_modules_layer_modules_13_modules_activation_min_val.item()
        l_self_modules_layer_modules_13_modules_activation_min_val = None
        item_59 = l_self_modules_layer_modules_13_modules_activation_max_val.item()
        l_self_modules_layer_modules_13_modules_activation_max_val = None
        features_59 = torch.nn.functional.hardtanh(features_58, item_58, item_59, False)
        features_58 = item_58 = item_59 = None
        features_60 = torch._C._nn.pad(features_59, (1, 1, 1, 1), "constant", 0.0)
        features_59 = None
        features_61 = torch.conv2d(
            features_60,
            l_self_modules_layer_modules_14_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        features_60 = (
            l_self_modules_layer_modules_14_modules_convolution_parameters_weight_
        ) = None
        item_60 = l_self_modules_layer_modules_14_modules_normalization_momentum.item()
        l_self_modules_layer_modules_14_modules_normalization_momentum = None
        item_61 = l_self_modules_layer_modules_14_modules_normalization_eps.item()
        l_self_modules_layer_modules_14_modules_normalization_eps = None
        features_62 = torch.nn.functional.batch_norm(
            features_61,
            l_self_modules_layer_modules_14_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_14_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_14_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_14_modules_normalization_parameters_bias_,
            False,
            item_60,
            item_61,
        )
        features_61 = (
            l_self_modules_layer_modules_14_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_14_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_14_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_14_modules_normalization_parameters_bias_
        ) = item_60 = item_61 = None
        item_62 = l_self_modules_layer_modules_14_modules_activation_min_val.item()
        l_self_modules_layer_modules_14_modules_activation_min_val = None
        item_63 = l_self_modules_layer_modules_14_modules_activation_max_val.item()
        l_self_modules_layer_modules_14_modules_activation_max_val = None
        features_63 = torch.nn.functional.hardtanh(features_62, item_62, item_63, False)
        features_62 = item_62 = item_63 = None
        features_64 = torch._C._nn.pad(features_63, (0, 0, 0, 0), "constant", 0.0)
        features_63 = None
        features_65 = torch.conv2d(
            features_64,
            l_self_modules_layer_modules_15_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_64 = (
            l_self_modules_layer_modules_15_modules_convolution_parameters_weight_
        ) = None
        item_64 = l_self_modules_layer_modules_15_modules_normalization_momentum.item()
        l_self_modules_layer_modules_15_modules_normalization_momentum = None
        item_65 = l_self_modules_layer_modules_15_modules_normalization_eps.item()
        l_self_modules_layer_modules_15_modules_normalization_eps = None
        features_66 = torch.nn.functional.batch_norm(
            features_65,
            l_self_modules_layer_modules_15_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_15_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_15_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_15_modules_normalization_parameters_bias_,
            False,
            item_64,
            item_65,
        )
        features_65 = (
            l_self_modules_layer_modules_15_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_15_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_15_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_15_modules_normalization_parameters_bias_
        ) = item_64 = item_65 = None
        item_66 = l_self_modules_layer_modules_15_modules_activation_min_val.item()
        l_self_modules_layer_modules_15_modules_activation_min_val = None
        item_67 = l_self_modules_layer_modules_15_modules_activation_max_val.item()
        l_self_modules_layer_modules_15_modules_activation_max_val = None
        features_67 = torch.nn.functional.hardtanh(features_66, item_66, item_67, False)
        features_66 = item_66 = item_67 = None
        features_68 = torch._C._nn.pad(features_67, (1, 1, 1, 1), "constant", 0.0)
        features_67 = None
        features_69 = torch.conv2d(
            features_68,
            l_self_modules_layer_modules_16_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        features_68 = (
            l_self_modules_layer_modules_16_modules_convolution_parameters_weight_
        ) = None
        item_68 = l_self_modules_layer_modules_16_modules_normalization_momentum.item()
        l_self_modules_layer_modules_16_modules_normalization_momentum = None
        item_69 = l_self_modules_layer_modules_16_modules_normalization_eps.item()
        l_self_modules_layer_modules_16_modules_normalization_eps = None
        features_70 = torch.nn.functional.batch_norm(
            features_69,
            l_self_modules_layer_modules_16_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_16_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_16_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_16_modules_normalization_parameters_bias_,
            False,
            item_68,
            item_69,
        )
        features_69 = (
            l_self_modules_layer_modules_16_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_16_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_16_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_16_modules_normalization_parameters_bias_
        ) = item_68 = item_69 = None
        item_70 = l_self_modules_layer_modules_16_modules_activation_min_val.item()
        l_self_modules_layer_modules_16_modules_activation_min_val = None
        item_71 = l_self_modules_layer_modules_16_modules_activation_max_val.item()
        l_self_modules_layer_modules_16_modules_activation_max_val = None
        features_71 = torch.nn.functional.hardtanh(features_70, item_70, item_71, False)
        features_70 = item_70 = item_71 = None
        features_72 = torch._C._nn.pad(features_71, (0, 0, 0, 0), "constant", 0.0)
        features_71 = None
        features_73 = torch.conv2d(
            features_72,
            l_self_modules_layer_modules_17_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_72 = (
            l_self_modules_layer_modules_17_modules_convolution_parameters_weight_
        ) = None
        item_72 = l_self_modules_layer_modules_17_modules_normalization_momentum.item()
        l_self_modules_layer_modules_17_modules_normalization_momentum = None
        item_73 = l_self_modules_layer_modules_17_modules_normalization_eps.item()
        l_self_modules_layer_modules_17_modules_normalization_eps = None
        features_74 = torch.nn.functional.batch_norm(
            features_73,
            l_self_modules_layer_modules_17_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_17_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_17_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_17_modules_normalization_parameters_bias_,
            False,
            item_72,
            item_73,
        )
        features_73 = (
            l_self_modules_layer_modules_17_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_17_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_17_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_17_modules_normalization_parameters_bias_
        ) = item_72 = item_73 = None
        item_74 = l_self_modules_layer_modules_17_modules_activation_min_val.item()
        l_self_modules_layer_modules_17_modules_activation_min_val = None
        item_75 = l_self_modules_layer_modules_17_modules_activation_max_val.item()
        l_self_modules_layer_modules_17_modules_activation_max_val = None
        features_75 = torch.nn.functional.hardtanh(features_74, item_74, item_75, False)
        features_74 = item_74 = item_75 = None
        features_76 = torch._C._nn.pad(features_75, (1, 1, 1, 1), "constant", 0.0)
        features_75 = None
        features_77 = torch.conv2d(
            features_76,
            l_self_modules_layer_modules_18_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        features_76 = (
            l_self_modules_layer_modules_18_modules_convolution_parameters_weight_
        ) = None
        item_76 = l_self_modules_layer_modules_18_modules_normalization_momentum.item()
        l_self_modules_layer_modules_18_modules_normalization_momentum = None
        item_77 = l_self_modules_layer_modules_18_modules_normalization_eps.item()
        l_self_modules_layer_modules_18_modules_normalization_eps = None
        features_78 = torch.nn.functional.batch_norm(
            features_77,
            l_self_modules_layer_modules_18_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_18_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_18_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_18_modules_normalization_parameters_bias_,
            False,
            item_76,
            item_77,
        )
        features_77 = (
            l_self_modules_layer_modules_18_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_18_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_18_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_18_modules_normalization_parameters_bias_
        ) = item_76 = item_77 = None
        item_78 = l_self_modules_layer_modules_18_modules_activation_min_val.item()
        l_self_modules_layer_modules_18_modules_activation_min_val = None
        item_79 = l_self_modules_layer_modules_18_modules_activation_max_val.item()
        l_self_modules_layer_modules_18_modules_activation_max_val = None
        features_79 = torch.nn.functional.hardtanh(features_78, item_78, item_79, False)
        features_78 = item_78 = item_79 = None
        features_80 = torch._C._nn.pad(features_79, (0, 0, 0, 0), "constant", 0.0)
        features_79 = None
        features_81 = torch.conv2d(
            features_80,
            l_self_modules_layer_modules_19_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_80 = (
            l_self_modules_layer_modules_19_modules_convolution_parameters_weight_
        ) = None
        item_80 = l_self_modules_layer_modules_19_modules_normalization_momentum.item()
        l_self_modules_layer_modules_19_modules_normalization_momentum = None
        item_81 = l_self_modules_layer_modules_19_modules_normalization_eps.item()
        l_self_modules_layer_modules_19_modules_normalization_eps = None
        features_82 = torch.nn.functional.batch_norm(
            features_81,
            l_self_modules_layer_modules_19_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_19_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_19_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_19_modules_normalization_parameters_bias_,
            False,
            item_80,
            item_81,
        )
        features_81 = (
            l_self_modules_layer_modules_19_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_19_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_19_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_19_modules_normalization_parameters_bias_
        ) = item_80 = item_81 = None
        item_82 = l_self_modules_layer_modules_19_modules_activation_min_val.item()
        l_self_modules_layer_modules_19_modules_activation_min_val = None
        item_83 = l_self_modules_layer_modules_19_modules_activation_max_val.item()
        l_self_modules_layer_modules_19_modules_activation_max_val = None
        features_83 = torch.nn.functional.hardtanh(features_82, item_82, item_83, False)
        features_82 = item_82 = item_83 = None
        features_84 = torch._C._nn.pad(features_83, (1, 1, 1, 1), "constant", 0.0)
        features_83 = None
        features_85 = torch.conv2d(
            features_84,
            l_self_modules_layer_modules_20_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        features_84 = (
            l_self_modules_layer_modules_20_modules_convolution_parameters_weight_
        ) = None
        item_84 = l_self_modules_layer_modules_20_modules_normalization_momentum.item()
        l_self_modules_layer_modules_20_modules_normalization_momentum = None
        item_85 = l_self_modules_layer_modules_20_modules_normalization_eps.item()
        l_self_modules_layer_modules_20_modules_normalization_eps = None
        features_86 = torch.nn.functional.batch_norm(
            features_85,
            l_self_modules_layer_modules_20_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_20_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_20_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_20_modules_normalization_parameters_bias_,
            False,
            item_84,
            item_85,
        )
        features_85 = (
            l_self_modules_layer_modules_20_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_20_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_20_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_20_modules_normalization_parameters_bias_
        ) = item_84 = item_85 = None
        item_86 = l_self_modules_layer_modules_20_modules_activation_min_val.item()
        l_self_modules_layer_modules_20_modules_activation_min_val = None
        item_87 = l_self_modules_layer_modules_20_modules_activation_max_val.item()
        l_self_modules_layer_modules_20_modules_activation_max_val = None
        features_87 = torch.nn.functional.hardtanh(features_86, item_86, item_87, False)
        features_86 = item_86 = item_87 = None
        features_88 = torch._C._nn.pad(features_87, (0, 0, 0, 0), "constant", 0.0)
        features_87 = None
        features_89 = torch.conv2d(
            features_88,
            l_self_modules_layer_modules_21_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_88 = (
            l_self_modules_layer_modules_21_modules_convolution_parameters_weight_
        ) = None
        item_88 = l_self_modules_layer_modules_21_modules_normalization_momentum.item()
        l_self_modules_layer_modules_21_modules_normalization_momentum = None
        item_89 = l_self_modules_layer_modules_21_modules_normalization_eps.item()
        l_self_modules_layer_modules_21_modules_normalization_eps = None
        features_90 = torch.nn.functional.batch_norm(
            features_89,
            l_self_modules_layer_modules_21_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_21_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_21_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_21_modules_normalization_parameters_bias_,
            False,
            item_88,
            item_89,
        )
        features_89 = (
            l_self_modules_layer_modules_21_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_21_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_21_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_21_modules_normalization_parameters_bias_
        ) = item_88 = item_89 = None
        item_90 = l_self_modules_layer_modules_21_modules_activation_min_val.item()
        l_self_modules_layer_modules_21_modules_activation_min_val = None
        item_91 = l_self_modules_layer_modules_21_modules_activation_max_val.item()
        l_self_modules_layer_modules_21_modules_activation_max_val = None
        features_91 = torch.nn.functional.hardtanh(features_90, item_90, item_91, False)
        features_90 = item_90 = item_91 = None
        features_92 = torch._C._nn.pad(features_91, (0, 1, 0, 1), "constant", 0.0)
        features_91 = None
        features_93 = torch.conv2d(
            features_92,
            l_self_modules_layer_modules_22_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            128,
        )
        features_92 = (
            l_self_modules_layer_modules_22_modules_convolution_parameters_weight_
        ) = None
        item_92 = l_self_modules_layer_modules_22_modules_normalization_momentum.item()
        l_self_modules_layer_modules_22_modules_normalization_momentum = None
        item_93 = l_self_modules_layer_modules_22_modules_normalization_eps.item()
        l_self_modules_layer_modules_22_modules_normalization_eps = None
        features_94 = torch.nn.functional.batch_norm(
            features_93,
            l_self_modules_layer_modules_22_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_22_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_22_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_22_modules_normalization_parameters_bias_,
            False,
            item_92,
            item_93,
        )
        features_93 = (
            l_self_modules_layer_modules_22_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_22_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_22_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_22_modules_normalization_parameters_bias_
        ) = item_92 = item_93 = None
        item_94 = l_self_modules_layer_modules_22_modules_activation_min_val.item()
        l_self_modules_layer_modules_22_modules_activation_min_val = None
        item_95 = l_self_modules_layer_modules_22_modules_activation_max_val.item()
        l_self_modules_layer_modules_22_modules_activation_max_val = None
        features_95 = torch.nn.functional.hardtanh(features_94, item_94, item_95, False)
        features_94 = item_94 = item_95 = None
        features_96 = torch._C._nn.pad(features_95, (0, 0, 0, 0), "constant", 0.0)
        features_95 = None
        features_97 = torch.conv2d(
            features_96,
            l_self_modules_layer_modules_23_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_96 = (
            l_self_modules_layer_modules_23_modules_convolution_parameters_weight_
        ) = None
        item_96 = l_self_modules_layer_modules_23_modules_normalization_momentum.item()
        l_self_modules_layer_modules_23_modules_normalization_momentum = None
        item_97 = l_self_modules_layer_modules_23_modules_normalization_eps.item()
        l_self_modules_layer_modules_23_modules_normalization_eps = None
        features_98 = torch.nn.functional.batch_norm(
            features_97,
            l_self_modules_layer_modules_23_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_23_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_23_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_23_modules_normalization_parameters_bias_,
            False,
            item_96,
            item_97,
        )
        features_97 = (
            l_self_modules_layer_modules_23_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_23_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_23_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_23_modules_normalization_parameters_bias_
        ) = item_96 = item_97 = None
        item_98 = l_self_modules_layer_modules_23_modules_activation_min_val.item()
        l_self_modules_layer_modules_23_modules_activation_min_val = None
        item_99 = l_self_modules_layer_modules_23_modules_activation_max_val.item()
        l_self_modules_layer_modules_23_modules_activation_max_val = None
        features_99 = torch.nn.functional.hardtanh(features_98, item_98, item_99, False)
        features_98 = item_98 = item_99 = None
        features_100 = torch._C._nn.pad(features_99, (1, 1, 1, 1), "constant", 0.0)
        features_99 = None
        features_101 = torch.conv2d(
            features_100,
            l_self_modules_layer_modules_24_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        features_100 = (
            l_self_modules_layer_modules_24_modules_convolution_parameters_weight_
        ) = None
        item_100 = l_self_modules_layer_modules_24_modules_normalization_momentum.item()
        l_self_modules_layer_modules_24_modules_normalization_momentum = None
        item_101 = l_self_modules_layer_modules_24_modules_normalization_eps.item()
        l_self_modules_layer_modules_24_modules_normalization_eps = None
        features_102 = torch.nn.functional.batch_norm(
            features_101,
            l_self_modules_layer_modules_24_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_24_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_24_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_24_modules_normalization_parameters_bias_,
            False,
            item_100,
            item_101,
        )
        features_101 = (
            l_self_modules_layer_modules_24_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_24_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_24_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_24_modules_normalization_parameters_bias_
        ) = item_100 = item_101 = None
        item_102 = l_self_modules_layer_modules_24_modules_activation_min_val.item()
        l_self_modules_layer_modules_24_modules_activation_min_val = None
        item_103 = l_self_modules_layer_modules_24_modules_activation_max_val.item()
        l_self_modules_layer_modules_24_modules_activation_max_val = None
        features_103 = torch.nn.functional.hardtanh(
            features_102, item_102, item_103, False
        )
        features_102 = item_102 = item_103 = None
        features_104 = torch._C._nn.pad(features_103, (0, 0, 0, 0), "constant", 0.0)
        features_103 = None
        features_105 = torch.conv2d(
            features_104,
            l_self_modules_layer_modules_25_modules_convolution_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        features_104 = (
            l_self_modules_layer_modules_25_modules_convolution_parameters_weight_
        ) = None
        item_104 = l_self_modules_layer_modules_25_modules_normalization_momentum.item()
        l_self_modules_layer_modules_25_modules_normalization_momentum = None
        item_105 = l_self_modules_layer_modules_25_modules_normalization_eps.item()
        l_self_modules_layer_modules_25_modules_normalization_eps = None
        features_106 = torch.nn.functional.batch_norm(
            features_105,
            l_self_modules_layer_modules_25_modules_normalization_buffers_running_mean_,
            l_self_modules_layer_modules_25_modules_normalization_buffers_running_var_,
            l_self_modules_layer_modules_25_modules_normalization_parameters_weight_,
            l_self_modules_layer_modules_25_modules_normalization_parameters_bias_,
            False,
            item_104,
            item_105,
        )
        features_105 = (
            l_self_modules_layer_modules_25_modules_normalization_buffers_running_mean_
        ) = (
            l_self_modules_layer_modules_25_modules_normalization_buffers_running_var_
        ) = (
            l_self_modules_layer_modules_25_modules_normalization_parameters_weight_
        ) = (
            l_self_modules_layer_modules_25_modules_normalization_parameters_bias_
        ) = item_104 = item_105 = None
        item_106 = l_self_modules_layer_modules_25_modules_activation_min_val.item()
        l_self_modules_layer_modules_25_modules_activation_min_val = None
        item_107 = l_self_modules_layer_modules_25_modules_activation_max_val.item()
        l_self_modules_layer_modules_25_modules_activation_max_val = None
        features_107 = torch.nn.functional.hardtanh(
            features_106, item_106, item_107, False
        )
        features_106 = item_106 = item_107 = None
        adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(
            features_107, (1, 1)
        )
        pooled_output = torch.flatten(adaptive_avg_pool2d, start_dim=1)
        adaptive_avg_pool2d = None
        return (features_107, pooled_output)
