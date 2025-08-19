import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s99: torch.SymInt,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embeddings_modules_padding_value: torch.Tensor,
        L_self_modules_embeddings_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_batchnorm_momentum: torch.Tensor,
        L_self_modules_embeddings_modules_batchnorm_buffers_running_mean_: torch.Tensor,
        L_self_modules_embeddings_modules_batchnorm_buffers_running_var_: torch.Tensor,
        L_self_modules_embeddings_modules_batchnorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_batchnorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_batchnorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_pad_value: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_eps: torch.Tensor,
        L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_dropout_p: torch.Tensor,
        L_self_modules_encoder_modules_top_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_top_bn_momentum: torch.Tensor,
        L_self_modules_encoder_modules_top_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_encoder_modules_top_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_encoder_modules_top_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_top_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_top_bn_eps: torch.Tensor,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embeddings_modules_padding_value = (
            L_self_modules_embeddings_modules_padding_value
        )
        l_self_modules_embeddings_modules_convolution_parameters_weight_ = (
            L_self_modules_embeddings_modules_convolution_parameters_weight_
        )
        l_self_modules_embeddings_modules_batchnorm_momentum = (
            L_self_modules_embeddings_modules_batchnorm_momentum
        )
        l_self_modules_embeddings_modules_batchnorm_buffers_running_mean_ = (
            L_self_modules_embeddings_modules_batchnorm_buffers_running_mean_
        )
        l_self_modules_embeddings_modules_batchnorm_buffers_running_var_ = (
            L_self_modules_embeddings_modules_batchnorm_buffers_running_var_
        )
        l_self_modules_embeddings_modules_batchnorm_parameters_weight_ = (
            L_self_modules_embeddings_modules_batchnorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_batchnorm_parameters_bias_ = (
            L_self_modules_embeddings_modules_batchnorm_parameters_bias_
        )
        l_self_modules_embeddings_modules_batchnorm_eps = (
            L_self_modules_embeddings_modules_batchnorm_eps
        )
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_pad_value = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_pad_value
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_momentum = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_eps = L_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_eps
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_momentum = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_momentum
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_eps = L_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_eps
        l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_conv_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_conv_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_momentum = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_momentum
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_mean_ = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_mean_
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_var_ = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_var_
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_weight_ = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_weight_
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_bias_ = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_bias_
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_eps = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_eps
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_dropout_p = L_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_dropout_p
        l_self_modules_encoder_modules_top_conv_parameters_weight_ = (
            L_self_modules_encoder_modules_top_conv_parameters_weight_
        )
        l_self_modules_encoder_modules_top_bn_momentum = (
            L_self_modules_encoder_modules_top_bn_momentum
        )
        l_self_modules_encoder_modules_top_bn_buffers_running_mean_ = (
            L_self_modules_encoder_modules_top_bn_buffers_running_mean_
        )
        l_self_modules_encoder_modules_top_bn_buffers_running_var_ = (
            L_self_modules_encoder_modules_top_bn_buffers_running_var_
        )
        l_self_modules_encoder_modules_top_bn_parameters_weight_ = (
            L_self_modules_encoder_modules_top_bn_parameters_weight_
        )
        l_self_modules_encoder_modules_top_bn_parameters_bias_ = (
            L_self_modules_encoder_modules_top_bn_parameters_bias_
        )
        l_self_modules_encoder_modules_top_bn_eps = (
            L_self_modules_encoder_modules_top_bn_eps
        )
        item = l_self_modules_embeddings_modules_padding_value.item()
        l_self_modules_embeddings_modules_padding_value = None
        features = torch._C._nn.pad(l_pixel_values_, (0, 1, 0, 1), "constant", item)
        l_pixel_values_ = item = None
        features_1 = torch.conv2d(
            features,
            l_self_modules_embeddings_modules_convolution_parameters_weight_,
            None,
            (2, 2),
            "valid",
            (1, 1),
            1,
        )
        features = (
            l_self_modules_embeddings_modules_convolution_parameters_weight_
        ) = None
        item_1 = l_self_modules_embeddings_modules_batchnorm_momentum.item()
        l_self_modules_embeddings_modules_batchnorm_momentum = None
        item_2 = l_self_modules_embeddings_modules_batchnorm_eps.item()
        l_self_modules_embeddings_modules_batchnorm_eps = None
        features_2 = torch.nn.functional.batch_norm(
            features_1,
            l_self_modules_embeddings_modules_batchnorm_buffers_running_mean_,
            l_self_modules_embeddings_modules_batchnorm_buffers_running_var_,
            l_self_modules_embeddings_modules_batchnorm_parameters_weight_,
            l_self_modules_embeddings_modules_batchnorm_parameters_bias_,
            False,
            item_1,
            item_2,
        )
        features_1 = (
            l_self_modules_embeddings_modules_batchnorm_buffers_running_mean_
        ) = (
            l_self_modules_embeddings_modules_batchnorm_buffers_running_var_
        ) = (
            l_self_modules_embeddings_modules_batchnorm_parameters_weight_
        ) = (
            l_self_modules_embeddings_modules_batchnorm_parameters_bias_
        ) = item_1 = item_2 = None
        features_3 = torch._C._nn.gelu(features_2)
        features_2 = None
        hidden_states = torch.conv2d(
            features_3,
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            64,
        )
        features_3 = l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_3 = (
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_4 = (
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_1 = torch.nn.functional.batch_norm(
            hidden_states,
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_3,
            item_4,
        )
        hidden_states = l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_0_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_3) = (
            item_4
        ) = None
        hidden_states_2 = torch._C._nn.gelu(hidden_states_1)
        hidden_states_1 = None
        hidden_states_3 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_2, 1)
        hidden_states_4 = torch.conv2d(
            hidden_states_3,
            l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_3 = l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_5 = torch._C._nn.gelu(hidden_states_4)
        hidden_states_4 = None
        hidden_states_6 = torch.conv2d(
            hidden_states_5,
            l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_5 = l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_0_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_7 = torch.sigmoid(hidden_states_6)
        hidden_states_6 = None
        hidden_states_8 = torch.mul(hidden_states_2, hidden_states_7)
        hidden_states_2 = hidden_states_7 = None
        hidden_states_9 = torch.conv2d(
            hidden_states_8,
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_8 = l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_5 = (
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_6 = (
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_10 = torch.nn.functional.batch_norm(
            hidden_states_9,
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_5,
            item_6,
        )
        hidden_states_9 = l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_0_modules_projection_modules_project_bn_parameters_bias_ = (item_5) = (
            item_6
        ) = None
        hidden_states_11 = torch.conv2d(
            hidden_states_10,
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            32,
        )
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (
            None
        )
        item_7 = (
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_8 = (
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_12 = torch.nn.functional.batch_norm(
            hidden_states_11,
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_7,
            item_8,
        )
        hidden_states_11 = l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_1_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_7) = (
            item_8
        ) = None
        hidden_states_13 = torch._C._nn.gelu(hidden_states_12)
        hidden_states_12 = None
        hidden_states_14 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_13, 1)
        hidden_states_15 = torch.conv2d(
            hidden_states_14,
            l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_14 = l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_16 = torch._C._nn.gelu(hidden_states_15)
        hidden_states_15 = None
        hidden_states_17 = torch.conv2d(
            hidden_states_16,
            l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_16 = l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_1_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_18 = torch.sigmoid(hidden_states_17)
        hidden_states_17 = None
        hidden_states_19 = torch.mul(hidden_states_13, hidden_states_18)
        hidden_states_13 = hidden_states_18 = None
        hidden_states_20 = torch.conv2d(
            hidden_states_19,
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_19 = l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_9 = (
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_10 = (
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_21 = torch.nn.functional.batch_norm(
            hidden_states_20,
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_9,
            item_10,
        )
        hidden_states_20 = l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_project_bn_parameters_bias_ = (item_9) = (
            item_10
        ) = None
        item_11 = (
            l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_1_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_22 = torch.nn.functional.dropout(
            hidden_states_21, item_11, False, False
        )
        hidden_states_21 = item_11 = None
        hidden_states_23 = hidden_states_22 + hidden_states_10
        hidden_states_22 = hidden_states_10 = None
        hidden_states_24 = torch.conv2d(
            hidden_states_23,
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            32,
        )
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (
            None
        )
        item_12 = (
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_13 = (
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_25 = torch.nn.functional.batch_norm(
            hidden_states_24,
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_12,
            item_13,
        )
        hidden_states_24 = l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_2_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_12) = (
            item_13
        ) = None
        hidden_states_26 = torch._C._nn.gelu(hidden_states_25)
        hidden_states_25 = None
        hidden_states_27 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_26, 1)
        hidden_states_28 = torch.conv2d(
            hidden_states_27,
            l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_27 = l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_29 = torch._C._nn.gelu(hidden_states_28)
        hidden_states_28 = None
        hidden_states_30 = torch.conv2d(
            hidden_states_29,
            l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_29 = l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_2_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_31 = torch.sigmoid(hidden_states_30)
        hidden_states_30 = None
        hidden_states_32 = torch.mul(hidden_states_26, hidden_states_31)
        hidden_states_26 = hidden_states_31 = None
        hidden_states_33 = torch.conv2d(
            hidden_states_32,
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_32 = l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_14 = (
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_15 = (
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_34 = torch.nn.functional.batch_norm(
            hidden_states_33,
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_14,
            item_15,
        )
        hidden_states_33 = l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_project_bn_parameters_bias_ = (item_14) = (
            item_15
        ) = None
        item_16 = (
            l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_2_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_35 = torch.nn.functional.dropout(
            hidden_states_34, item_16, False, False
        )
        hidden_states_34 = item_16 = None
        hidden_states_36 = hidden_states_35 + hidden_states_23
        hidden_states_35 = hidden_states_23 = None
        hidden_states_37 = torch.conv2d(
            hidden_states_36,
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            32,
        )
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (
            None
        )
        item_17 = (
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_18 = (
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_38 = torch.nn.functional.batch_norm(
            hidden_states_37,
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_17,
            item_18,
        )
        hidden_states_37 = l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_3_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_17) = (
            item_18
        ) = None
        hidden_states_39 = torch._C._nn.gelu(hidden_states_38)
        hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_39, 1)
        hidden_states_41 = torch.conv2d(
            hidden_states_40,
            l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_40 = l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_42 = torch._C._nn.gelu(hidden_states_41)
        hidden_states_41 = None
        hidden_states_43 = torch.conv2d(
            hidden_states_42,
            l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_42 = l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_3_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_44 = torch.sigmoid(hidden_states_43)
        hidden_states_43 = None
        hidden_states_45 = torch.mul(hidden_states_39, hidden_states_44)
        hidden_states_39 = hidden_states_44 = None
        hidden_states_46 = torch.conv2d(
            hidden_states_45,
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_45 = l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_19 = (
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_20 = (
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_47 = torch.nn.functional.batch_norm(
            hidden_states_46,
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_19,
            item_20,
        )
        hidden_states_46 = l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_project_bn_parameters_bias_ = (item_19) = (
            item_20
        ) = None
        item_21 = (
            l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_3_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_48 = torch.nn.functional.dropout(
            hidden_states_47, item_21, False, False
        )
        hidden_states_47 = item_21 = None
        hidden_states_49 = hidden_states_48 + hidden_states_36
        hidden_states_48 = hidden_states_36 = None
        hidden_states_50 = torch.conv2d(
            hidden_states_49,
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_49 = l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_conv_parameters_weight_ = (None)
        item_22 = (
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_23 = (
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_51 = torch.nn.functional.batch_norm(
            hidden_states_50,
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_22,
            item_23,
        )
        hidden_states_50 = l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_4_modules_expansion_modules_expand_bn_parameters_bias_ = (item_22) = (
            item_23
        ) = None
        hidden_states_52 = torch._C._nn.gelu(hidden_states_51)
        hidden_states_51 = None
        hidden_states_53 = torch.conv2d(
            hidden_states_52,
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            192,
        )
        hidden_states_52 = l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_24 = (
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_25 = (
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_54 = torch.nn.functional.batch_norm(
            hidden_states_53,
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        hidden_states_53 = l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_4_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_24) = (
            item_25
        ) = None
        hidden_states_55 = torch._C._nn.gelu(hidden_states_54)
        hidden_states_54 = None
        hidden_states_56 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_55, 1)
        hidden_states_57 = torch.conv2d(
            hidden_states_56,
            l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_56 = l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_58 = torch._C._nn.gelu(hidden_states_57)
        hidden_states_57 = None
        hidden_states_59 = torch.conv2d(
            hidden_states_58,
            l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_58 = l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_4_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_60 = torch.sigmoid(hidden_states_59)
        hidden_states_59 = None
        hidden_states_61 = torch.mul(hidden_states_55, hidden_states_60)
        hidden_states_55 = hidden_states_60 = None
        hidden_states_62 = torch.conv2d(
            hidden_states_61,
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_61 = l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_26 = (
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_27 = (
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_63 = torch.nn.functional.batch_norm(
            hidden_states_62,
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_26,
            item_27,
        )
        hidden_states_62 = l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_4_modules_projection_modules_project_bn_parameters_bias_ = (item_26) = (
            item_27
        ) = None
        hidden_states_64 = torch.conv2d(
            hidden_states_63,
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_28 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_29 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_65 = torch.nn.functional.batch_norm(
            hidden_states_64,
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_28,
            item_29,
        )
        hidden_states_64 = l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_5_modules_expansion_modules_expand_bn_parameters_bias_ = (item_28) = (
            item_29
        ) = None
        hidden_states_66 = torch._C._nn.gelu(hidden_states_65)
        hidden_states_65 = None
        hidden_states_67 = torch.conv2d(
            hidden_states_66,
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            288,
        )
        hidden_states_66 = l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_30 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_31 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_68 = torch.nn.functional.batch_norm(
            hidden_states_67,
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_30,
            item_31,
        )
        hidden_states_67 = l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_5_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_30) = (
            item_31
        ) = None
        hidden_states_69 = torch._C._nn.gelu(hidden_states_68)
        hidden_states_68 = None
        hidden_states_70 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_69, 1)
        hidden_states_71 = torch.conv2d(
            hidden_states_70,
            l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_70 = l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_72 = torch._C._nn.gelu(hidden_states_71)
        hidden_states_71 = None
        hidden_states_73 = torch.conv2d(
            hidden_states_72,
            l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_72 = l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_5_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_74 = torch.sigmoid(hidden_states_73)
        hidden_states_73 = None
        hidden_states_75 = torch.mul(hidden_states_69, hidden_states_74)
        hidden_states_69 = hidden_states_74 = None
        hidden_states_76 = torch.conv2d(
            hidden_states_75,
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_75 = l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_32 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_33 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_77 = torch.nn.functional.batch_norm(
            hidden_states_76,
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_32,
            item_33,
        )
        hidden_states_76 = l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_project_bn_parameters_bias_ = (item_32) = (
            item_33
        ) = None
        item_34 = (
            l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_5_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_78 = torch.nn.functional.dropout(
            hidden_states_77, item_34, False, False
        )
        hidden_states_77 = item_34 = None
        hidden_states_79 = hidden_states_78 + hidden_states_63
        hidden_states_78 = hidden_states_63 = None
        hidden_states_80 = torch.conv2d(
            hidden_states_79,
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_35 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_36 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_81 = torch.nn.functional.batch_norm(
            hidden_states_80,
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_35,
            item_36,
        )
        hidden_states_80 = l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_6_modules_expansion_modules_expand_bn_parameters_bias_ = (item_35) = (
            item_36
        ) = None
        hidden_states_82 = torch._C._nn.gelu(hidden_states_81)
        hidden_states_81 = None
        hidden_states_83 = torch.conv2d(
            hidden_states_82,
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            288,
        )
        hidden_states_82 = l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_37 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_38 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_84 = torch.nn.functional.batch_norm(
            hidden_states_83,
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_37,
            item_38,
        )
        hidden_states_83 = l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_6_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_37) = (
            item_38
        ) = None
        hidden_states_85 = torch._C._nn.gelu(hidden_states_84)
        hidden_states_84 = None
        hidden_states_86 = torch.nn.functional.adaptive_avg_pool2d(hidden_states_85, 1)
        hidden_states_87 = torch.conv2d(
            hidden_states_86,
            l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_86 = l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_88 = torch._C._nn.gelu(hidden_states_87)
        hidden_states_87 = None
        hidden_states_89 = torch.conv2d(
            hidden_states_88,
            l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_88 = l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_6_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_90 = torch.sigmoid(hidden_states_89)
        hidden_states_89 = None
        hidden_states_91 = torch.mul(hidden_states_85, hidden_states_90)
        hidden_states_85 = hidden_states_90 = None
        hidden_states_92 = torch.conv2d(
            hidden_states_91,
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_91 = l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_39 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_40 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_93 = torch.nn.functional.batch_norm(
            hidden_states_92,
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_39,
            item_40,
        )
        hidden_states_92 = l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_project_bn_parameters_bias_ = (item_39) = (
            item_40
        ) = None
        item_41 = (
            l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_6_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_94 = torch.nn.functional.dropout(
            hidden_states_93, item_41, False, False
        )
        hidden_states_93 = item_41 = None
        hidden_states_95 = hidden_states_94 + hidden_states_79
        hidden_states_94 = hidden_states_79 = None
        hidden_states_96 = torch.conv2d(
            hidden_states_95,
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_42 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_43 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_97 = torch.nn.functional.batch_norm(
            hidden_states_96,
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_42,
            item_43,
        )
        hidden_states_96 = l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_7_modules_expansion_modules_expand_bn_parameters_bias_ = (item_42) = (
            item_43
        ) = None
        hidden_states_98 = torch._C._nn.gelu(hidden_states_97)
        hidden_states_97 = None
        hidden_states_99 = torch.conv2d(
            hidden_states_98,
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            288,
        )
        hidden_states_98 = l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_44 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_45 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_100 = torch.nn.functional.batch_norm(
            hidden_states_99,
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_44,
            item_45,
        )
        hidden_states_99 = l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_7_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_44) = (
            item_45
        ) = None
        hidden_states_101 = torch._C._nn.gelu(hidden_states_100)
        hidden_states_100 = None
        hidden_states_102 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_101, 1
        )
        hidden_states_103 = torch.conv2d(
            hidden_states_102,
            l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_102 = l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_104 = torch._C._nn.gelu(hidden_states_103)
        hidden_states_103 = None
        hidden_states_105 = torch.conv2d(
            hidden_states_104,
            l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_104 = l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_7_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_106 = torch.sigmoid(hidden_states_105)
        hidden_states_105 = None
        hidden_states_107 = torch.mul(hidden_states_101, hidden_states_106)
        hidden_states_101 = hidden_states_106 = None
        hidden_states_108 = torch.conv2d(
            hidden_states_107,
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_107 = l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_46 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_47 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_109 = torch.nn.functional.batch_norm(
            hidden_states_108,
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_46,
            item_47,
        )
        hidden_states_108 = l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_project_bn_parameters_bias_ = (item_46) = (
            item_47
        ) = None
        item_48 = (
            l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_7_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_110 = torch.nn.functional.dropout(
            hidden_states_109, item_48, False, False
        )
        hidden_states_109 = item_48 = None
        hidden_states_111 = hidden_states_110 + hidden_states_95
        hidden_states_110 = hidden_states_95 = None
        hidden_states_112 = torch.conv2d(
            hidden_states_111,
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_111 = l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_conv_parameters_weight_ = (None)
        item_49 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_50 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_113 = torch.nn.functional.batch_norm(
            hidden_states_112,
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_49,
            item_50,
        )
        hidden_states_112 = l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_8_modules_expansion_modules_expand_bn_parameters_bias_ = (item_49) = (
            item_50
        ) = None
        hidden_states_114 = torch._C._nn.gelu(hidden_states_113)
        hidden_states_113 = None
        item_51 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_pad_value.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_pad_value = (
            None
        )
        hidden_states_115 = torch._C._nn.pad(
            hidden_states_114, (1, 2, 1, 2), "constant", item_51
        )
        hidden_states_114 = item_51 = None
        hidden_states_116 = torch.conv2d(
            hidden_states_115,
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (2, 2),
            "valid",
            (1, 1),
            288,
        )
        hidden_states_115 = l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_52 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_53 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_117 = torch.nn.functional.batch_norm(
            hidden_states_116,
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_52,
            item_53,
        )
        hidden_states_116 = l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_8_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_52) = (
            item_53
        ) = None
        hidden_states_118 = torch._C._nn.gelu(hidden_states_117)
        hidden_states_117 = None
        hidden_states_119 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_118, 1
        )
        hidden_states_120 = torch.conv2d(
            hidden_states_119,
            l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_119 = l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_121 = torch._C._nn.gelu(hidden_states_120)
        hidden_states_120 = None
        hidden_states_122 = torch.conv2d(
            hidden_states_121,
            l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_121 = l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_8_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_123 = torch.sigmoid(hidden_states_122)
        hidden_states_122 = None
        hidden_states_124 = torch.mul(hidden_states_118, hidden_states_123)
        hidden_states_118 = hidden_states_123 = None
        hidden_states_125 = torch.conv2d(
            hidden_states_124,
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_124 = l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_54 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_55 = (
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_126 = torch.nn.functional.batch_norm(
            hidden_states_125,
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_54,
            item_55,
        )
        hidden_states_125 = l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_8_modules_projection_modules_project_bn_parameters_bias_ = (item_54) = (
            item_55
        ) = None
        hidden_states_127 = torch.conv2d(
            hidden_states_126,
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_56 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_57 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_128 = torch.nn.functional.batch_norm(
            hidden_states_127,
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_56,
            item_57,
        )
        hidden_states_127 = l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_9_modules_expansion_modules_expand_bn_parameters_bias_ = (item_56) = (
            item_57
        ) = None
        hidden_states_129 = torch._C._nn.gelu(hidden_states_128)
        hidden_states_128 = None
        hidden_states_130 = torch.conv2d(
            hidden_states_129,
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            480,
        )
        hidden_states_129 = l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_58 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_59 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_131 = torch.nn.functional.batch_norm(
            hidden_states_130,
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_58,
            item_59,
        )
        hidden_states_130 = l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_9_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_58) = (
            item_59
        ) = None
        hidden_states_132 = torch._C._nn.gelu(hidden_states_131)
        hidden_states_131 = None
        hidden_states_133 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_132, 1
        )
        hidden_states_134 = torch.conv2d(
            hidden_states_133,
            l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_133 = l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_135 = torch._C._nn.gelu(hidden_states_134)
        hidden_states_134 = None
        hidden_states_136 = torch.conv2d(
            hidden_states_135,
            l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_135 = l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_9_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_137 = torch.sigmoid(hidden_states_136)
        hidden_states_136 = None
        hidden_states_138 = torch.mul(hidden_states_132, hidden_states_137)
        hidden_states_132 = hidden_states_137 = None
        hidden_states_139 = torch.conv2d(
            hidden_states_138,
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_138 = l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_60 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_61 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_140 = torch.nn.functional.batch_norm(
            hidden_states_139,
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_60,
            item_61,
        )
        hidden_states_139 = l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_project_bn_parameters_bias_ = (item_60) = (
            item_61
        ) = None
        item_62 = (
            l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_9_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_141 = torch.nn.functional.dropout(
            hidden_states_140, item_62, False, False
        )
        hidden_states_140 = item_62 = None
        hidden_states_142 = hidden_states_141 + hidden_states_126
        hidden_states_141 = hidden_states_126 = None
        hidden_states_143 = torch.conv2d(
            hidden_states_142,
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_63 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_64 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_144 = torch.nn.functional.batch_norm(
            hidden_states_143,
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_63,
            item_64,
        )
        hidden_states_143 = l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_10_modules_expansion_modules_expand_bn_parameters_bias_ = (item_63) = (
            item_64
        ) = None
        hidden_states_145 = torch._C._nn.gelu(hidden_states_144)
        hidden_states_144 = None
        hidden_states_146 = torch.conv2d(
            hidden_states_145,
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            480,
        )
        hidden_states_145 = l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_65 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_66 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_147 = torch.nn.functional.batch_norm(
            hidden_states_146,
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_65,
            item_66,
        )
        hidden_states_146 = l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_10_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_65) = (
            item_66
        ) = None
        hidden_states_148 = torch._C._nn.gelu(hidden_states_147)
        hidden_states_147 = None
        hidden_states_149 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_148, 1
        )
        hidden_states_150 = torch.conv2d(
            hidden_states_149,
            l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_149 = l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_151 = torch._C._nn.gelu(hidden_states_150)
        hidden_states_150 = None
        hidden_states_152 = torch.conv2d(
            hidden_states_151,
            l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_151 = l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_10_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_153 = torch.sigmoid(hidden_states_152)
        hidden_states_152 = None
        hidden_states_154 = torch.mul(hidden_states_148, hidden_states_153)
        hidden_states_148 = hidden_states_153 = None
        hidden_states_155 = torch.conv2d(
            hidden_states_154,
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_154 = l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_67 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_68 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_156 = torch.nn.functional.batch_norm(
            hidden_states_155,
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_67,
            item_68,
        )
        hidden_states_155 = l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_project_bn_parameters_bias_ = (item_67) = (
            item_68
        ) = None
        item_69 = (
            l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_10_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_157 = torch.nn.functional.dropout(
            hidden_states_156, item_69, False, False
        )
        hidden_states_156 = item_69 = None
        hidden_states_158 = hidden_states_157 + hidden_states_142
        hidden_states_157 = hidden_states_142 = None
        hidden_states_159 = torch.conv2d(
            hidden_states_158,
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_70 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_71 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_160 = torch.nn.functional.batch_norm(
            hidden_states_159,
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_70,
            item_71,
        )
        hidden_states_159 = l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_11_modules_expansion_modules_expand_bn_parameters_bias_ = (item_70) = (
            item_71
        ) = None
        hidden_states_161 = torch._C._nn.gelu(hidden_states_160)
        hidden_states_160 = None
        hidden_states_162 = torch.conv2d(
            hidden_states_161,
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            480,
        )
        hidden_states_161 = l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_72 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_73 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_163 = torch.nn.functional.batch_norm(
            hidden_states_162,
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_72,
            item_73,
        )
        hidden_states_162 = l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_11_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_72) = (
            item_73
        ) = None
        hidden_states_164 = torch._C._nn.gelu(hidden_states_163)
        hidden_states_163 = None
        hidden_states_165 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_164, 1
        )
        hidden_states_166 = torch.conv2d(
            hidden_states_165,
            l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_165 = l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_167 = torch._C._nn.gelu(hidden_states_166)
        hidden_states_166 = None
        hidden_states_168 = torch.conv2d(
            hidden_states_167,
            l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_167 = l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_11_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_169 = torch.sigmoid(hidden_states_168)
        hidden_states_168 = None
        hidden_states_170 = torch.mul(hidden_states_164, hidden_states_169)
        hidden_states_164 = hidden_states_169 = None
        hidden_states_171 = torch.conv2d(
            hidden_states_170,
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_170 = l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_74 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_75 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_172 = torch.nn.functional.batch_norm(
            hidden_states_171,
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_74,
            item_75,
        )
        hidden_states_171 = l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_project_bn_parameters_bias_ = (item_74) = (
            item_75
        ) = None
        item_76 = (
            l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_11_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_173 = torch.nn.functional.dropout(
            hidden_states_172, item_76, False, False
        )
        hidden_states_172 = item_76 = None
        hidden_states_174 = hidden_states_173 + hidden_states_158
        hidden_states_173 = hidden_states_158 = None
        hidden_states_175 = torch.conv2d(
            hidden_states_174,
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_77 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_78 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_176 = torch.nn.functional.batch_norm(
            hidden_states_175,
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_77,
            item_78,
        )
        hidden_states_175 = l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_12_modules_expansion_modules_expand_bn_parameters_bias_ = (item_77) = (
            item_78
        ) = None
        hidden_states_177 = torch._C._nn.gelu(hidden_states_176)
        hidden_states_176 = None
        hidden_states_178 = torch.conv2d(
            hidden_states_177,
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            480,
        )
        hidden_states_177 = l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_79 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_80 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_179 = torch.nn.functional.batch_norm(
            hidden_states_178,
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_79,
            item_80,
        )
        hidden_states_178 = l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_12_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_79) = (
            item_80
        ) = None
        hidden_states_180 = torch._C._nn.gelu(hidden_states_179)
        hidden_states_179 = None
        hidden_states_181 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_180, 1
        )
        hidden_states_182 = torch.conv2d(
            hidden_states_181,
            l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_181 = l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_183 = torch._C._nn.gelu(hidden_states_182)
        hidden_states_182 = None
        hidden_states_184 = torch.conv2d(
            hidden_states_183,
            l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_183 = l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_12_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_185 = torch.sigmoid(hidden_states_184)
        hidden_states_184 = None
        hidden_states_186 = torch.mul(hidden_states_180, hidden_states_185)
        hidden_states_180 = hidden_states_185 = None
        hidden_states_187 = torch.conv2d(
            hidden_states_186,
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_186 = l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_81 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_82 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_188 = torch.nn.functional.batch_norm(
            hidden_states_187,
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_81,
            item_82,
        )
        hidden_states_187 = l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_project_bn_parameters_bias_ = (item_81) = (
            item_82
        ) = None
        item_83 = (
            l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_12_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_189 = torch.nn.functional.dropout(
            hidden_states_188, item_83, False, False
        )
        hidden_states_188 = item_83 = None
        hidden_states_190 = hidden_states_189 + hidden_states_174
        hidden_states_189 = hidden_states_174 = None
        hidden_states_191 = torch.conv2d(
            hidden_states_190,
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_84 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_85 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_192 = torch.nn.functional.batch_norm(
            hidden_states_191,
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_84,
            item_85,
        )
        hidden_states_191 = l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_13_modules_expansion_modules_expand_bn_parameters_bias_ = (item_84) = (
            item_85
        ) = None
        hidden_states_193 = torch._C._nn.gelu(hidden_states_192)
        hidden_states_192 = None
        hidden_states_194 = torch.conv2d(
            hidden_states_193,
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            480,
        )
        hidden_states_193 = l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_86 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_87 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_195 = torch.nn.functional.batch_norm(
            hidden_states_194,
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_86,
            item_87,
        )
        hidden_states_194 = l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_13_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_86) = (
            item_87
        ) = None
        hidden_states_196 = torch._C._nn.gelu(hidden_states_195)
        hidden_states_195 = None
        hidden_states_197 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_196, 1
        )
        hidden_states_198 = torch.conv2d(
            hidden_states_197,
            l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_197 = l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_199 = torch._C._nn.gelu(hidden_states_198)
        hidden_states_198 = None
        hidden_states_200 = torch.conv2d(
            hidden_states_199,
            l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_199 = l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_13_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_201 = torch.sigmoid(hidden_states_200)
        hidden_states_200 = None
        hidden_states_202 = torch.mul(hidden_states_196, hidden_states_201)
        hidden_states_196 = hidden_states_201 = None
        hidden_states_203 = torch.conv2d(
            hidden_states_202,
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_202 = l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_88 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_89 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_204 = torch.nn.functional.batch_norm(
            hidden_states_203,
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_88,
            item_89,
        )
        hidden_states_203 = l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_project_bn_parameters_bias_ = (item_88) = (
            item_89
        ) = None
        item_90 = (
            l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_13_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_205 = torch.nn.functional.dropout(
            hidden_states_204, item_90, False, False
        )
        hidden_states_204 = item_90 = None
        hidden_states_206 = hidden_states_205 + hidden_states_190
        hidden_states_205 = hidden_states_190 = None
        hidden_states_207 = torch.conv2d(
            hidden_states_206,
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_conv_parameters_weight_ = (
            None
        )
        item_91 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_momentum = (
            None
        )
        item_92 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_eps = (
            None
        )
        hidden_states_208 = torch.nn.functional.batch_norm(
            hidden_states_207,
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_bias_,
            False,
            item_91,
            item_92,
        )
        hidden_states_207 = l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_14_modules_expansion_modules_expand_bn_parameters_bias_ = (item_91) = (
            item_92
        ) = None
        hidden_states_209 = torch._C._nn.gelu(hidden_states_208)
        hidden_states_208 = None
        hidden_states_210 = torch.conv2d(
            hidden_states_209,
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            480,
        )
        hidden_states_209 = l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_conv_parameters_weight_ = (None)
        item_93 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_momentum = (
            None
        )
        item_94 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_eps = (
            None
        )
        hidden_states_211 = torch.nn.functional.batch_norm(
            hidden_states_210,
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_,
            False,
            item_93,
            item_94,
        )
        hidden_states_210 = l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_14_modules_depthwise_conv_modules_depthwise_norm_parameters_bias_ = (item_93) = (
            item_94
        ) = None
        hidden_states_212 = torch._C._nn.gelu(hidden_states_211)
        hidden_states_211 = None
        hidden_states_213 = torch.nn.functional.adaptive_avg_pool2d(
            hidden_states_212, 1
        )
        hidden_states_214 = torch.conv2d(
            hidden_states_213,
            l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_213 = l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_reduce_parameters_bias_ = (None)
        hidden_states_215 = torch._C._nn.gelu(hidden_states_214)
        hidden_states_214 = None
        hidden_states_216 = torch.conv2d(
            hidden_states_215,
            l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_bias_,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_215 = l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_14_modules_squeeze_excite_modules_expand_parameters_bias_ = (None)
        hidden_states_217 = torch.sigmoid(hidden_states_216)
        hidden_states_216 = None
        hidden_states_218 = torch.mul(hidden_states_212, hidden_states_217)
        hidden_states_212 = hidden_states_217 = None
        hidden_states_219 = torch.conv2d(
            hidden_states_218,
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_218 = l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_conv_parameters_weight_ = (None)
        item_95 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_momentum.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_momentum = (
            None
        )
        item_96 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_eps.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_eps = (
            None
        )
        hidden_states_220 = torch.nn.functional.batch_norm(
            hidden_states_219,
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_var_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_weight_,
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_bias_,
            False,
            item_95,
            item_96,
        )
        hidden_states_219 = l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_mean_ = l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_buffers_running_var_ = l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_weight_ = l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_project_bn_parameters_bias_ = (item_95) = (
            item_96
        ) = None
        item_97 = (
            l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_dropout_p.item()
        )
        l_self_modules_encoder_modules_blocks_modules_14_modules_projection_modules_dropout_p = (
            None
        )
        hidden_states_221 = torch.nn.functional.dropout(
            hidden_states_220, item_97, False, False
        )
        hidden_states_220 = item_97 = None
        hidden_states_222 = hidden_states_221 + hidden_states_206
        hidden_states_221 = hidden_states_206 = None
        hidden_states_223 = torch.conv2d(
            hidden_states_222,
            l_self_modules_encoder_modules_top_conv_parameters_weight_,
            None,
            (1, 1),
            "same",
            (1, 1),
            1,
        )
        hidden_states_222 = (
            l_self_modules_encoder_modules_top_conv_parameters_weight_
        ) = None
        item_98 = l_self_modules_encoder_modules_top_bn_momentum.item()
        l_self_modules_encoder_modules_top_bn_momentum = None
        item_99 = l_self_modules_encoder_modules_top_bn_eps.item()
        l_self_modules_encoder_modules_top_bn_eps = None
        hidden_states_224 = torch.nn.functional.batch_norm(
            hidden_states_223,
            l_self_modules_encoder_modules_top_bn_buffers_running_mean_,
            l_self_modules_encoder_modules_top_bn_buffers_running_var_,
            l_self_modules_encoder_modules_top_bn_parameters_weight_,
            l_self_modules_encoder_modules_top_bn_parameters_bias_,
            False,
            item_98,
            item_99,
        )
        hidden_states_223 = (
            l_self_modules_encoder_modules_top_bn_buffers_running_mean_
        ) = (
            l_self_modules_encoder_modules_top_bn_buffers_running_var_
        ) = (
            l_self_modules_encoder_modules_top_bn_parameters_weight_
        ) = (
            l_self_modules_encoder_modules_top_bn_parameters_bias_
        ) = item_98 = item_99 = None
        hidden_states_225 = torch._C._nn.gelu(hidden_states_224)
        hidden_states_224 = None
        pooled_output = torch._C._nn.avg_pool2d(
            hidden_states_225, 2560, 2560, 0, True, True, None
        )
        floordiv = s99 // 4
        s99 = None
        sym_sum = torch.sym_sum([-1, floordiv])
        floordiv = None
        floordiv_1 = sym_sum // 2560
        sym_sum = None
        sym_sum_1 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_1 = None
        pooled_output_1 = pooled_output.reshape((1, 2560))
        pooled_output = None
        return (hidden_states_225, pooled_output_1)
