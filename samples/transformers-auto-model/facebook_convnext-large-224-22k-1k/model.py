import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s99: torch.SymInt,
        s4: torch.SymInt,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embeddings_modules_patch_embeddings_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_patch_embeddings_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_layernorm_eps: torch.Tensor,
        L_self_modules_embeddings_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embeddings_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_parameters_layer_scale_parameter_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layernorm_eps: torch.Tensor,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embeddings_modules_patch_embeddings_parameters_weight_ = (
            L_self_modules_embeddings_modules_patch_embeddings_parameters_weight_
        )
        l_self_modules_embeddings_modules_patch_embeddings_parameters_bias_ = (
            L_self_modules_embeddings_modules_patch_embeddings_parameters_bias_
        )
        l_self_modules_embeddings_modules_layernorm_eps = (
            L_self_modules_embeddings_modules_layernorm_eps
        )
        l_self_modules_embeddings_modules_layernorm_parameters_weight_ = (
            L_self_modules_embeddings_modules_layernorm_parameters_weight_
        )
        l_self_modules_embeddings_modules_layernorm_parameters_bias_ = (
            L_self_modules_embeddings_modules_layernorm_parameters_bias_
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_eps = L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_eps = L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_eps = L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_parameters_layer_scale_parameter_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_parameters_layer_scale_parameter_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_parameters_layer_scale_parameter_
        l_self_modules_layernorm_parameters_weight_ = (
            L_self_modules_layernorm_parameters_weight_
        )
        l_self_modules_layernorm_parameters_bias_ = (
            L_self_modules_layernorm_parameters_bias_
        )
        l_self_modules_layernorm_eps = L_self_modules_layernorm_eps
        embeddings = torch.conv2d(
            l_pixel_values_,
            l_self_modules_embeddings_modules_patch_embeddings_parameters_weight_,
            l_self_modules_embeddings_modules_patch_embeddings_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_pixel_values_ = (
            l_self_modules_embeddings_modules_patch_embeddings_parameters_weight_
        ) = l_self_modules_embeddings_modules_patch_embeddings_parameters_bias_ = None
        x = embeddings.float()
        embeddings = None
        u = x.mean(1, keepdim=True)
        sub = x - u
        pow_1 = sub.pow(2)
        sub = None
        s = pow_1.mean(1, keepdim=True)
        pow_1 = None
        sub_1 = x - u
        x = u = None
        item = l_self_modules_embeddings_modules_layernorm_eps.item()
        l_self_modules_embeddings_modules_layernorm_eps = None
        add = s + item
        s = item = None
        sqrt = torch.sqrt(add)
        add = None
        x_1 = sub_1 / sqrt
        sub_1 = sqrt = None
        x_2 = x_1.to(dtype=torch.float32)
        x_1 = None
        getitem_4 = l_self_modules_embeddings_modules_layernorm_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_embeddings_modules_layernorm_parameters_weight_ = None
        mul = getitem_4 * x_2
        getitem_4 = x_2 = None
        getitem_5 = l_self_modules_embeddings_modules_layernorm_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_embeddings_modules_layernorm_parameters_bias_ = None
        x_3 = mul + getitem_5
        mul = getitem_5 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_bias_ = (None)
        x_5 = x_4.permute(0, 2, 3, 1)
        x_4 = None
        item_1 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_eps = (
            None
        )
        x_6 = torch.nn.functional.layer_norm(
            x_5,
            (192,),
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_bias_,
            item_1,
        )
        x_5 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_bias_ = (item_1) = (
            None
        )
        x_7 = torch._C._nn.linear(
            x_6,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_bias_,
        )
        x_6 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = (None)
        x_8 = torch._C._nn.gelu(x_7)
        x_7 = None
        x_9 = torch._C._nn.linear(
            x_8,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_bias_,
        )
        x_8 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = (None)
        x_10 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_parameters_layer_scale_parameter_
            * x_9
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_parameters_layer_scale_parameter_ = (
            x_9
        ) = None
        x_11 = x_10.permute(0, 3, 1, 2)
        x_10 = None
        x_12 = x_3 + x_11
        x_3 = x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_bias_ = (None)
        x_14 = x_13.permute(0, 2, 3, 1)
        x_13 = None
        item_2 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_eps = (
            None
        )
        x_15 = torch.nn.functional.layer_norm(
            x_14,
            (192,),
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_bias_,
            item_2,
        )
        x_14 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_bias_ = (item_2) = (
            None
        )
        x_16 = torch._C._nn.linear(
            x_15,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_bias_,
        )
        x_15 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = (None)
        x_17 = torch._C._nn.gelu(x_16)
        x_16 = None
        x_18 = torch._C._nn.linear(
            x_17,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_bias_,
        )
        x_17 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = (None)
        x_19 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_parameters_layer_scale_parameter_
            * x_18
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_parameters_layer_scale_parameter_ = (
            x_18
        ) = None
        x_20 = x_19.permute(0, 3, 1, 2)
        x_19 = None
        x_21 = x_12 + x_20
        x_12 = x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_dwconv_parameters_bias_ = (None)
        x_23 = x_22.permute(0, 2, 3, 1)
        x_22 = None
        item_3 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_eps = (
            None
        )
        x_24 = torch.nn.functional.layer_norm(
            x_23,
            (192,),
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_bias_,
            item_3,
        )
        x_23 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_layernorm_parameters_bias_ = (item_3) = (
            None
        )
        x_25 = torch._C._nn.linear(
            x_24,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_bias_,
        )
        x_24 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = (None)
        x_26 = torch._C._nn.gelu(x_25)
        x_25 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_bias_,
        )
        x_26 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = (None)
        x_28 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_parameters_layer_scale_parameter_
            * x_27
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_2_parameters_layer_scale_parameter_ = (
            x_27
        ) = None
        x_29 = x_28.permute(0, 3, 1, 2)
        x_28 = None
        x_30 = x_21 + x_29
        x_21 = x_29 = None
        x_31 = x_30.float()
        x_30 = None
        u_1 = x_31.mean(1, keepdim=True)
        sub_2 = x_31 - u_1
        pow_2 = sub_2.pow(2)
        sub_2 = None
        s_1 = pow_2.mean(1, keepdim=True)
        pow_2 = None
        sub_3 = x_31 - u_1
        x_31 = u_1 = None
        item_4 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_eps = (
            None
        )
        add_5 = s_1 + item_4
        s_1 = item_4 = None
        sqrt_1 = torch.sqrt(add_5)
        add_5 = None
        x_32 = sub_3 / sqrt_1
        sub_3 = sqrt_1 = None
        x_33 = x_32.to(dtype=torch.float32)
        x_32 = None
        getitem_6 = l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_weight_ = (
            None
        )
        mul_4 = getitem_6 * x_33
        getitem_6 = x_33 = None
        getitem_7 = l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_bias_ = (
            None
        )
        x_34 = mul_4 + getitem_7
        mul_4 = getitem_7 = None
        input_1 = torch.conv2d(
            x_34,
            l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_bias_ = (None)
        x_35 = torch.conv2d(
            input_1,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_bias_ = (None)
        x_36 = x_35.permute(0, 2, 3, 1)
        x_35 = None
        item_5 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_eps = (
            None
        )
        x_37 = torch.nn.functional.layer_norm(
            x_36,
            (384,),
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_bias_,
            item_5,
        )
        x_36 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_bias_ = (item_5) = (
            None
        )
        x_38 = torch._C._nn.linear(
            x_37,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_bias_,
        )
        x_37 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = (None)
        x_39 = torch._C._nn.gelu(x_38)
        x_38 = None
        x_40 = torch._C._nn.linear(
            x_39,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_bias_,
        )
        x_39 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = (None)
        x_41 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_parameters_layer_scale_parameter_
            * x_40
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_parameters_layer_scale_parameter_ = (
            x_40
        ) = None
        x_42 = x_41.permute(0, 3, 1, 2)
        x_41 = None
        x_43 = input_1 + x_42
        input_1 = x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_bias_ = (None)
        x_45 = x_44.permute(0, 2, 3, 1)
        x_44 = None
        item_6 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_eps = (
            None
        )
        x_46 = torch.nn.functional.layer_norm(
            x_45,
            (384,),
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_bias_,
            item_6,
        )
        x_45 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_bias_ = (item_6) = (
            None
        )
        x_47 = torch._C._nn.linear(
            x_46,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_bias_,
        )
        x_46 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = (None)
        x_48 = torch._C._nn.gelu(x_47)
        x_47 = None
        x_49 = torch._C._nn.linear(
            x_48,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_bias_,
        )
        x_48 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = (None)
        x_50 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_parameters_layer_scale_parameter_
            * x_49
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_parameters_layer_scale_parameter_ = (
            x_49
        ) = None
        x_51 = x_50.permute(0, 3, 1, 2)
        x_50 = None
        x_52 = x_43 + x_51
        x_43 = x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_dwconv_parameters_bias_ = (None)
        x_54 = x_53.permute(0, 2, 3, 1)
        x_53 = None
        item_7 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_eps = (
            None
        )
        x_55 = torch.nn.functional.layer_norm(
            x_54,
            (384,),
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_bias_,
            item_7,
        )
        x_54 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_layernorm_parameters_bias_ = (item_7) = (
            None
        )
        x_56 = torch._C._nn.linear(
            x_55,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_bias_,
        )
        x_55 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = (None)
        x_57 = torch._C._nn.gelu(x_56)
        x_56 = None
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_bias_,
        )
        x_57 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = (None)
        x_59 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_parameters_layer_scale_parameter_
            * x_58
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_2_parameters_layer_scale_parameter_ = (
            x_58
        ) = None
        x_60 = x_59.permute(0, 3, 1, 2)
        x_59 = None
        x_61 = x_52 + x_60
        x_52 = x_60 = None
        x_62 = x_61.float()
        x_61 = None
        u_2 = x_62.mean(1, keepdim=True)
        sub_4 = x_62 - u_2
        pow_3 = sub_4.pow(2)
        sub_4 = None
        s_2 = pow_3.mean(1, keepdim=True)
        pow_3 = None
        sub_5 = x_62 - u_2
        x_62 = u_2 = None
        item_8 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_eps = (
            None
        )
        add_10 = s_2 + item_8
        s_2 = item_8 = None
        sqrt_2 = torch.sqrt(add_10)
        add_10 = None
        x_63 = sub_5 / sqrt_2
        sub_5 = sqrt_2 = None
        x_64 = x_63.to(dtype=torch.float32)
        x_63 = None
        getitem_8 = l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_weight_ = (
            None
        )
        mul_8 = getitem_8 * x_64
        getitem_8 = x_64 = None
        getitem_9 = l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_bias_ = (
            None
        )
        x_65 = mul_8 + getitem_9
        mul_8 = getitem_9 = None
        input_2 = torch.conv2d(
            x_65,
            l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            input_2,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_bias_ = (None)
        x_67 = x_66.permute(0, 2, 3, 1)
        x_66 = None
        item_9 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_eps = (
            None
        )
        x_68 = torch.nn.functional.layer_norm(
            x_67,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_bias_,
            item_9,
        )
        x_67 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_bias_ = (item_9) = (
            None
        )
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_bias_,
        )
        x_68 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = (None)
        x_70 = torch._C._nn.gelu(x_69)
        x_69 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_bias_,
        )
        x_70 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = (None)
        x_72 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_parameters_layer_scale_parameter_
            * x_71
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_parameters_layer_scale_parameter_ = (
            x_71
        ) = None
        x_73 = x_72.permute(0, 3, 1, 2)
        x_72 = None
        x_74 = input_2 + x_73
        input_2 = x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_bias_ = (None)
        x_76 = x_75.permute(0, 2, 3, 1)
        x_75 = None
        item_10 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_eps = (
            None
        )
        x_77 = torch.nn.functional.layer_norm(
            x_76,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_bias_,
            item_10,
        )
        x_76 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_bias_ = (item_10) = (
            None
        )
        x_78 = torch._C._nn.linear(
            x_77,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_bias_,
        )
        x_77 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = (None)
        x_79 = torch._C._nn.gelu(x_78)
        x_78 = None
        x_80 = torch._C._nn.linear(
            x_79,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_bias_,
        )
        x_79 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = (None)
        x_81 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_parameters_layer_scale_parameter_
            * x_80
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_parameters_layer_scale_parameter_ = (
            x_80
        ) = None
        x_82 = x_81.permute(0, 3, 1, 2)
        x_81 = None
        x_83 = x_74 + x_82
        x_74 = x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_bias_ = (None)
        x_85 = x_84.permute(0, 2, 3, 1)
        x_84 = None
        item_11 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_eps = (
            None
        )
        x_86 = torch.nn.functional.layer_norm(
            x_85,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_bias_,
            item_11,
        )
        x_85 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_bias_ = (item_11) = (
            None
        )
        x_87 = torch._C._nn.linear(
            x_86,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_bias_,
        )
        x_86 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = (None)
        x_88 = torch._C._nn.gelu(x_87)
        x_87 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_bias_,
        )
        x_88 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = (None)
        x_90 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_parameters_layer_scale_parameter_
            * x_89
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_parameters_layer_scale_parameter_ = (
            x_89
        ) = None
        x_91 = x_90.permute(0, 3, 1, 2)
        x_90 = None
        x_92 = x_83 + x_91
        x_83 = x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_dwconv_parameters_bias_ = (None)
        x_94 = x_93.permute(0, 2, 3, 1)
        x_93 = None
        item_12 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_eps = (
            None
        )
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_bias_,
            item_12,
        )
        x_94 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_layernorm_parameters_bias_ = (item_12) = (
            None
        )
        x_96 = torch._C._nn.linear(
            x_95,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_bias_,
        )
        x_95 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv1_parameters_bias_ = (None)
        x_97 = torch._C._nn.gelu(x_96)
        x_96 = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_bias_,
        )
        x_97 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_pwconv2_parameters_bias_ = (None)
        x_99 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_parameters_layer_scale_parameter_
            * x_98
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_parameters_layer_scale_parameter_ = (
            x_98
        ) = None
        x_100 = x_99.permute(0, 3, 1, 2)
        x_99 = None
        x_101 = x_92 + x_100
        x_92 = x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_dwconv_parameters_bias_ = (None)
        x_103 = x_102.permute(0, 2, 3, 1)
        x_102 = None
        item_13 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_eps = (
            None
        )
        x_104 = torch.nn.functional.layer_norm(
            x_103,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_bias_,
            item_13,
        )
        x_103 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_layernorm_parameters_bias_ = (item_13) = (
            None
        )
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_bias_,
        )
        x_104 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv1_parameters_bias_ = (None)
        x_106 = torch._C._nn.gelu(x_105)
        x_105 = None
        x_107 = torch._C._nn.linear(
            x_106,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_bias_,
        )
        x_106 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_pwconv2_parameters_bias_ = (None)
        x_108 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_parameters_layer_scale_parameter_
            * x_107
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_parameters_layer_scale_parameter_ = (
            x_107
        ) = None
        x_109 = x_108.permute(0, 3, 1, 2)
        x_108 = None
        x_110 = x_101 + x_109
        x_101 = x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_dwconv_parameters_bias_ = (None)
        x_112 = x_111.permute(0, 2, 3, 1)
        x_111 = None
        item_14 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_eps = (
            None
        )
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_bias_,
            item_14,
        )
        x_112 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_layernorm_parameters_bias_ = (item_14) = (
            None
        )
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_bias_,
        )
        x_113 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv1_parameters_bias_ = (None)
        x_115 = torch._C._nn.gelu(x_114)
        x_114 = None
        x_116 = torch._C._nn.linear(
            x_115,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_bias_,
        )
        x_115 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_pwconv2_parameters_bias_ = (None)
        x_117 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_parameters_layer_scale_parameter_
            * x_116
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_parameters_layer_scale_parameter_ = (
            x_116
        ) = None
        x_118 = x_117.permute(0, 3, 1, 2)
        x_117 = None
        x_119 = x_110 + x_118
        x_110 = x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_dwconv_parameters_bias_ = (None)
        x_121 = x_120.permute(0, 2, 3, 1)
        x_120 = None
        item_15 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_eps = (
            None
        )
        x_122 = torch.nn.functional.layer_norm(
            x_121,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_bias_,
            item_15,
        )
        x_121 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_layernorm_parameters_bias_ = (item_15) = (
            None
        )
        x_123 = torch._C._nn.linear(
            x_122,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_bias_,
        )
        x_122 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv1_parameters_bias_ = (None)
        x_124 = torch._C._nn.gelu(x_123)
        x_123 = None
        x_125 = torch._C._nn.linear(
            x_124,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_bias_,
        )
        x_124 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_pwconv2_parameters_bias_ = (None)
        x_126 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_parameters_layer_scale_parameter_
            * x_125
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_parameters_layer_scale_parameter_ = (
            x_125
        ) = None
        x_127 = x_126.permute(0, 3, 1, 2)
        x_126 = None
        x_128 = x_119 + x_127
        x_119 = x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_dwconv_parameters_bias_ = (None)
        x_130 = x_129.permute(0, 2, 3, 1)
        x_129 = None
        item_16 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_eps = (
            None
        )
        x_131 = torch.nn.functional.layer_norm(
            x_130,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_bias_,
            item_16,
        )
        x_130 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_layernorm_parameters_bias_ = (item_16) = (
            None
        )
        x_132 = torch._C._nn.linear(
            x_131,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_bias_,
        )
        x_131 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv1_parameters_bias_ = (None)
        x_133 = torch._C._nn.gelu(x_132)
        x_132 = None
        x_134 = torch._C._nn.linear(
            x_133,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_bias_,
        )
        x_133 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_pwconv2_parameters_bias_ = (None)
        x_135 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_parameters_layer_scale_parameter_
            * x_134
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_parameters_layer_scale_parameter_ = (
            x_134
        ) = None
        x_136 = x_135.permute(0, 3, 1, 2)
        x_135 = None
        x_137 = x_128 + x_136
        x_128 = x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_dwconv_parameters_bias_ = (None)
        x_139 = x_138.permute(0, 2, 3, 1)
        x_138 = None
        item_17 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_eps = (
            None
        )
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_bias_,
            item_17,
        )
        x_139 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_layernorm_parameters_bias_ = (item_17) = (
            None
        )
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_bias_,
        )
        x_140 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv1_parameters_bias_ = (None)
        x_142 = torch._C._nn.gelu(x_141)
        x_141 = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_bias_,
        )
        x_142 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_pwconv2_parameters_bias_ = (None)
        x_144 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_parameters_layer_scale_parameter_
            * x_143
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_parameters_layer_scale_parameter_ = (
            x_143
        ) = None
        x_145 = x_144.permute(0, 3, 1, 2)
        x_144 = None
        x_146 = x_137 + x_145
        x_137 = x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_dwconv_parameters_bias_ = (None)
        x_148 = x_147.permute(0, 2, 3, 1)
        x_147 = None
        item_18 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_eps = (
            None
        )
        x_149 = torch.nn.functional.layer_norm(
            x_148,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_bias_,
            item_18,
        )
        x_148 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_layernorm_parameters_bias_ = (item_18) = (
            None
        )
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_bias_,
        )
        x_149 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv1_parameters_bias_ = (None)
        x_151 = torch._C._nn.gelu(x_150)
        x_150 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_bias_,
        )
        x_151 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_pwconv2_parameters_bias_ = (None)
        x_153 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_parameters_layer_scale_parameter_
            * x_152
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_parameters_layer_scale_parameter_ = (
            x_152
        ) = None
        x_154 = x_153.permute(0, 3, 1, 2)
        x_153 = None
        x_155 = x_146 + x_154
        x_146 = x_154 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_dwconv_parameters_bias_ = (None)
        x_157 = x_156.permute(0, 2, 3, 1)
        x_156 = None
        item_19 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_eps = (
            None
        )
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_bias_,
            item_19,
        )
        x_157 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_layernorm_parameters_bias_ = (item_19) = (
            None
        )
        x_159 = torch._C._nn.linear(
            x_158,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_bias_,
        )
        x_158 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv1_parameters_bias_ = (None)
        x_160 = torch._C._nn.gelu(x_159)
        x_159 = None
        x_161 = torch._C._nn.linear(
            x_160,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_bias_,
        )
        x_160 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_pwconv2_parameters_bias_ = (None)
        x_162 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_parameters_layer_scale_parameter_
            * x_161
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_parameters_layer_scale_parameter_ = (
            x_161
        ) = None
        x_163 = x_162.permute(0, 3, 1, 2)
        x_162 = None
        x_164 = x_155 + x_163
        x_155 = x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_dwconv_parameters_bias_ = (None)
        x_166 = x_165.permute(0, 2, 3, 1)
        x_165 = None
        item_20 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_eps = (
            None
        )
        x_167 = torch.nn.functional.layer_norm(
            x_166,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_bias_,
            item_20,
        )
        x_166 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_layernorm_parameters_bias_ = (item_20) = (
            None
        )
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_bias_,
        )
        x_167 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv1_parameters_bias_ = (None)
        x_169 = torch._C._nn.gelu(x_168)
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_bias_,
        )
        x_169 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_pwconv2_parameters_bias_ = (None)
        x_171 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_parameters_layer_scale_parameter_
            * x_170
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_parameters_layer_scale_parameter_ = (
            x_170
        ) = None
        x_172 = x_171.permute(0, 3, 1, 2)
        x_171 = None
        x_173 = x_164 + x_172
        x_164 = x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_dwconv_parameters_bias_ = (None)
        x_175 = x_174.permute(0, 2, 3, 1)
        x_174 = None
        item_21 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_eps = (
            None
        )
        x_176 = torch.nn.functional.layer_norm(
            x_175,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_bias_,
            item_21,
        )
        x_175 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_layernorm_parameters_bias_ = (item_21) = (
            None
        )
        x_177 = torch._C._nn.linear(
            x_176,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_bias_,
        )
        x_176 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv1_parameters_bias_ = (None)
        x_178 = torch._C._nn.gelu(x_177)
        x_177 = None
        x_179 = torch._C._nn.linear(
            x_178,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_bias_,
        )
        x_178 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_pwconv2_parameters_bias_ = (None)
        x_180 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_parameters_layer_scale_parameter_
            * x_179
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_parameters_layer_scale_parameter_ = (
            x_179
        ) = None
        x_181 = x_180.permute(0, 3, 1, 2)
        x_180 = None
        x_182 = x_173 + x_181
        x_173 = x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_dwconv_parameters_bias_ = (None)
        x_184 = x_183.permute(0, 2, 3, 1)
        x_183 = None
        item_22 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_eps = (
            None
        )
        x_185 = torch.nn.functional.layer_norm(
            x_184,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_bias_,
            item_22,
        )
        x_184 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_layernorm_parameters_bias_ = (item_22) = (
            None
        )
        x_186 = torch._C._nn.linear(
            x_185,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_bias_,
        )
        x_185 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv1_parameters_bias_ = (None)
        x_187 = torch._C._nn.gelu(x_186)
        x_186 = None
        x_188 = torch._C._nn.linear(
            x_187,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_bias_,
        )
        x_187 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_pwconv2_parameters_bias_ = (None)
        x_189 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_parameters_layer_scale_parameter_
            * x_188
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_parameters_layer_scale_parameter_ = (
            x_188
        ) = None
        x_190 = x_189.permute(0, 3, 1, 2)
        x_189 = None
        x_191 = x_182 + x_190
        x_182 = x_190 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_dwconv_parameters_bias_ = (None)
        x_193 = x_192.permute(0, 2, 3, 1)
        x_192 = None
        item_23 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_eps = (
            None
        )
        x_194 = torch.nn.functional.layer_norm(
            x_193,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_bias_,
            item_23,
        )
        x_193 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_layernorm_parameters_bias_ = (item_23) = (
            None
        )
        x_195 = torch._C._nn.linear(
            x_194,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_bias_,
        )
        x_194 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv1_parameters_bias_ = (None)
        x_196 = torch._C._nn.gelu(x_195)
        x_195 = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_bias_,
        )
        x_196 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_pwconv2_parameters_bias_ = (None)
        x_198 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_parameters_layer_scale_parameter_
            * x_197
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_parameters_layer_scale_parameter_ = (
            x_197
        ) = None
        x_199 = x_198.permute(0, 3, 1, 2)
        x_198 = None
        x_200 = x_191 + x_199
        x_191 = x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_dwconv_parameters_bias_ = (None)
        x_202 = x_201.permute(0, 2, 3, 1)
        x_201 = None
        item_24 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_eps = (
            None
        )
        x_203 = torch.nn.functional.layer_norm(
            x_202,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_bias_,
            item_24,
        )
        x_202 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_layernorm_parameters_bias_ = (item_24) = (
            None
        )
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_bias_,
        )
        x_203 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv1_parameters_bias_ = (None)
        x_205 = torch._C._nn.gelu(x_204)
        x_204 = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_bias_,
        )
        x_205 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_pwconv2_parameters_bias_ = (None)
        x_207 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_parameters_layer_scale_parameter_
            * x_206
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_parameters_layer_scale_parameter_ = (
            x_206
        ) = None
        x_208 = x_207.permute(0, 3, 1, 2)
        x_207 = None
        x_209 = x_200 + x_208
        x_200 = x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_dwconv_parameters_bias_ = (None)
        x_211 = x_210.permute(0, 2, 3, 1)
        x_210 = None
        item_25 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_eps = (
            None
        )
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_bias_,
            item_25,
        )
        x_211 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_layernorm_parameters_bias_ = (item_25) = (
            None
        )
        x_213 = torch._C._nn.linear(
            x_212,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_bias_,
        )
        x_212 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv1_parameters_bias_ = (None)
        x_214 = torch._C._nn.gelu(x_213)
        x_213 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_bias_,
        )
        x_214 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_pwconv2_parameters_bias_ = (None)
        x_216 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_parameters_layer_scale_parameter_
            * x_215
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_parameters_layer_scale_parameter_ = (
            x_215
        ) = None
        x_217 = x_216.permute(0, 3, 1, 2)
        x_216 = None
        x_218 = x_209 + x_217
        x_209 = x_217 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_dwconv_parameters_bias_ = (None)
        x_220 = x_219.permute(0, 2, 3, 1)
        x_219 = None
        item_26 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_eps = (
            None
        )
        x_221 = torch.nn.functional.layer_norm(
            x_220,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_bias_,
            item_26,
        )
        x_220 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_layernorm_parameters_bias_ = (item_26) = (
            None
        )
        x_222 = torch._C._nn.linear(
            x_221,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_bias_,
        )
        x_221 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv1_parameters_bias_ = (None)
        x_223 = torch._C._nn.gelu(x_222)
        x_222 = None
        x_224 = torch._C._nn.linear(
            x_223,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_bias_,
        )
        x_223 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_pwconv2_parameters_bias_ = (None)
        x_225 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_parameters_layer_scale_parameter_
            * x_224
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_parameters_layer_scale_parameter_ = (
            x_224
        ) = None
        x_226 = x_225.permute(0, 3, 1, 2)
        x_225 = None
        x_227 = x_218 + x_226
        x_218 = x_226 = None
        x_228 = torch.conv2d(
            x_227,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_dwconv_parameters_bias_ = (None)
        x_229 = x_228.permute(0, 2, 3, 1)
        x_228 = None
        item_27 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_eps = (
            None
        )
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_bias_,
            item_27,
        )
        x_229 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_layernorm_parameters_bias_ = (item_27) = (
            None
        )
        x_231 = torch._C._nn.linear(
            x_230,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_bias_,
        )
        x_230 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv1_parameters_bias_ = (None)
        x_232 = torch._C._nn.gelu(x_231)
        x_231 = None
        x_233 = torch._C._nn.linear(
            x_232,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_bias_,
        )
        x_232 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_modules_pwconv2_parameters_bias_ = (None)
        x_234 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_parameters_layer_scale_parameter_
            * x_233
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_18_parameters_layer_scale_parameter_ = (
            x_233
        ) = None
        x_235 = x_234.permute(0, 3, 1, 2)
        x_234 = None
        x_236 = x_227 + x_235
        x_227 = x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_dwconv_parameters_bias_ = (None)
        x_238 = x_237.permute(0, 2, 3, 1)
        x_237 = None
        item_28 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_eps = (
            None
        )
        x_239 = torch.nn.functional.layer_norm(
            x_238,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_bias_,
            item_28,
        )
        x_238 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_layernorm_parameters_bias_ = (item_28) = (
            None
        )
        x_240 = torch._C._nn.linear(
            x_239,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_bias_,
        )
        x_239 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv1_parameters_bias_ = (None)
        x_241 = torch._C._nn.gelu(x_240)
        x_240 = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_bias_,
        )
        x_241 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_modules_pwconv2_parameters_bias_ = (None)
        x_243 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_parameters_layer_scale_parameter_
            * x_242
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_19_parameters_layer_scale_parameter_ = (
            x_242
        ) = None
        x_244 = x_243.permute(0, 3, 1, 2)
        x_243 = None
        x_245 = x_236 + x_244
        x_236 = x_244 = None
        x_246 = torch.conv2d(
            x_245,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_dwconv_parameters_bias_ = (None)
        x_247 = x_246.permute(0, 2, 3, 1)
        x_246 = None
        item_29 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_eps = (
            None
        )
        x_248 = torch.nn.functional.layer_norm(
            x_247,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_bias_,
            item_29,
        )
        x_247 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_layernorm_parameters_bias_ = (item_29) = (
            None
        )
        x_249 = torch._C._nn.linear(
            x_248,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_bias_,
        )
        x_248 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv1_parameters_bias_ = (None)
        x_250 = torch._C._nn.gelu(x_249)
        x_249 = None
        x_251 = torch._C._nn.linear(
            x_250,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_bias_,
        )
        x_250 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_modules_pwconv2_parameters_bias_ = (None)
        x_252 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_parameters_layer_scale_parameter_
            * x_251
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_20_parameters_layer_scale_parameter_ = (
            x_251
        ) = None
        x_253 = x_252.permute(0, 3, 1, 2)
        x_252 = None
        x_254 = x_245 + x_253
        x_245 = x_253 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_dwconv_parameters_bias_ = (None)
        x_256 = x_255.permute(0, 2, 3, 1)
        x_255 = None
        item_30 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_eps = (
            None
        )
        x_257 = torch.nn.functional.layer_norm(
            x_256,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_bias_,
            item_30,
        )
        x_256 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_layernorm_parameters_bias_ = (item_30) = (
            None
        )
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_bias_,
        )
        x_257 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv1_parameters_bias_ = (None)
        x_259 = torch._C._nn.gelu(x_258)
        x_258 = None
        x_260 = torch._C._nn.linear(
            x_259,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_bias_,
        )
        x_259 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_modules_pwconv2_parameters_bias_ = (None)
        x_261 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_parameters_layer_scale_parameter_
            * x_260
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_21_parameters_layer_scale_parameter_ = (
            x_260
        ) = None
        x_262 = x_261.permute(0, 3, 1, 2)
        x_261 = None
        x_263 = x_254 + x_262
        x_254 = x_262 = None
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_dwconv_parameters_bias_ = (None)
        x_265 = x_264.permute(0, 2, 3, 1)
        x_264 = None
        item_31 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_eps = (
            None
        )
        x_266 = torch.nn.functional.layer_norm(
            x_265,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_bias_,
            item_31,
        )
        x_265 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_layernorm_parameters_bias_ = (item_31) = (
            None
        )
        x_267 = torch._C._nn.linear(
            x_266,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_bias_,
        )
        x_266 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv1_parameters_bias_ = (None)
        x_268 = torch._C._nn.gelu(x_267)
        x_267 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_bias_,
        )
        x_268 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_modules_pwconv2_parameters_bias_ = (None)
        x_270 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_parameters_layer_scale_parameter_
            * x_269
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_22_parameters_layer_scale_parameter_ = (
            x_269
        ) = None
        x_271 = x_270.permute(0, 3, 1, 2)
        x_270 = None
        x_272 = x_263 + x_271
        x_263 = x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_dwconv_parameters_bias_ = (None)
        x_274 = x_273.permute(0, 2, 3, 1)
        x_273 = None
        item_32 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_eps = (
            None
        )
        x_275 = torch.nn.functional.layer_norm(
            x_274,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_bias_,
            item_32,
        )
        x_274 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_layernorm_parameters_bias_ = (item_32) = (
            None
        )
        x_276 = torch._C._nn.linear(
            x_275,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_bias_,
        )
        x_275 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv1_parameters_bias_ = (None)
        x_277 = torch._C._nn.gelu(x_276)
        x_276 = None
        x_278 = torch._C._nn.linear(
            x_277,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_bias_,
        )
        x_277 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_modules_pwconv2_parameters_bias_ = (None)
        x_279 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_parameters_layer_scale_parameter_
            * x_278
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_23_parameters_layer_scale_parameter_ = (
            x_278
        ) = None
        x_280 = x_279.permute(0, 3, 1, 2)
        x_279 = None
        x_281 = x_272 + x_280
        x_272 = x_280 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_dwconv_parameters_bias_ = (None)
        x_283 = x_282.permute(0, 2, 3, 1)
        x_282 = None
        item_33 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_eps = (
            None
        )
        x_284 = torch.nn.functional.layer_norm(
            x_283,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_bias_,
            item_33,
        )
        x_283 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_layernorm_parameters_bias_ = (item_33) = (
            None
        )
        x_285 = torch._C._nn.linear(
            x_284,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_bias_,
        )
        x_284 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv1_parameters_bias_ = (None)
        x_286 = torch._C._nn.gelu(x_285)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_bias_,
        )
        x_286 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_modules_pwconv2_parameters_bias_ = (None)
        x_288 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_parameters_layer_scale_parameter_
            * x_287
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_24_parameters_layer_scale_parameter_ = (
            x_287
        ) = None
        x_289 = x_288.permute(0, 3, 1, 2)
        x_288 = None
        x_290 = x_281 + x_289
        x_281 = x_289 = None
        x_291 = torch.conv2d(
            x_290,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_dwconv_parameters_bias_ = (None)
        x_292 = x_291.permute(0, 2, 3, 1)
        x_291 = None
        item_34 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_eps = (
            None
        )
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_bias_,
            item_34,
        )
        x_292 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_layernorm_parameters_bias_ = (item_34) = (
            None
        )
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_bias_,
        )
        x_293 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv1_parameters_bias_ = (None)
        x_295 = torch._C._nn.gelu(x_294)
        x_294 = None
        x_296 = torch._C._nn.linear(
            x_295,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_bias_,
        )
        x_295 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_modules_pwconv2_parameters_bias_ = (None)
        x_297 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_parameters_layer_scale_parameter_
            * x_296
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_25_parameters_layer_scale_parameter_ = (
            x_296
        ) = None
        x_298 = x_297.permute(0, 3, 1, 2)
        x_297 = None
        x_299 = x_290 + x_298
        x_290 = x_298 = None
        x_300 = torch.conv2d(
            x_299,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_dwconv_parameters_bias_ = (None)
        x_301 = x_300.permute(0, 2, 3, 1)
        x_300 = None
        item_35 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_eps = (
            None
        )
        x_302 = torch.nn.functional.layer_norm(
            x_301,
            (768,),
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_bias_,
            item_35,
        )
        x_301 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_layernorm_parameters_bias_ = (item_35) = (
            None
        )
        x_303 = torch._C._nn.linear(
            x_302,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_bias_,
        )
        x_302 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv1_parameters_bias_ = (None)
        x_304 = torch._C._nn.gelu(x_303)
        x_303 = None
        x_305 = torch._C._nn.linear(
            x_304,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_bias_,
        )
        x_304 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_modules_pwconv2_parameters_bias_ = (None)
        x_306 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_parameters_layer_scale_parameter_
            * x_305
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_26_parameters_layer_scale_parameter_ = (
            x_305
        ) = None
        x_307 = x_306.permute(0, 3, 1, 2)
        x_306 = None
        x_308 = x_299 + x_307
        x_299 = x_307 = None
        x_309 = x_308.float()
        x_308 = None
        u_3 = x_309.mean(1, keepdim=True)
        sub_6 = x_309 - u_3
        pow_4 = sub_6.pow(2)
        sub_6 = None
        s_3 = pow_4.mean(1, keepdim=True)
        pow_4 = None
        sub_7 = x_309 - u_3
        x_309 = u_3 = None
        item_36 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_eps = (
            None
        )
        add_39 = s_3 + item_36
        s_3 = item_36 = None
        sqrt_3 = torch.sqrt(add_39)
        add_39 = None
        x_310 = sub_7 / sqrt_3
        sub_7 = sqrt_3 = None
        x_311 = x_310.to(dtype=torch.float32)
        x_310 = None
        getitem_10 = l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_weight_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_weight_ = (
            None
        )
        mul_36 = getitem_10 * x_311
        getitem_10 = x_311 = None
        getitem_11 = l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_bias_[
            (slice(None, None, None), None, None)
        ]
        l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_bias_ = (
            None
        )
        x_312 = mul_36 + getitem_11
        mul_36 = getitem_11 = None
        input_3 = torch.conv2d(
            x_312,
            l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_312 = l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_bias_ = (None)
        x_313 = torch.conv2d(
            input_3,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1536,
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_bias_ = (None)
        x_314 = x_313.permute(0, 2, 3, 1)
        x_313 = None
        item_37 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_eps = (
            None
        )
        x_315 = torch.nn.functional.layer_norm(
            x_314,
            (1536,),
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_bias_,
            item_37,
        )
        x_314 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_bias_ = (item_37) = (
            None
        )
        x_316 = torch._C._nn.linear(
            x_315,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_bias_,
        )
        x_315 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_bias_ = (None)
        x_317 = torch._C._nn.gelu(x_316)
        x_316 = None
        x_318 = torch._C._nn.linear(
            x_317,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_bias_,
        )
        x_317 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_bias_ = (None)
        x_319 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_parameters_layer_scale_parameter_
            * x_318
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_parameters_layer_scale_parameter_ = (
            x_318
        ) = None
        x_320 = x_319.permute(0, 3, 1, 2)
        x_319 = None
        x_321 = input_3 + x_320
        input_3 = x_320 = None
        x_322 = torch.conv2d(
            x_321,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1536,
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_bias_ = (None)
        x_323 = x_322.permute(0, 2, 3, 1)
        x_322 = None
        item_38 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_eps = (
            None
        )
        x_324 = torch.nn.functional.layer_norm(
            x_323,
            (1536,),
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_bias_,
            item_38,
        )
        x_323 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_bias_ = (item_38) = (
            None
        )
        x_325 = torch._C._nn.linear(
            x_324,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_bias_,
        )
        x_324 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_bias_ = (None)
        x_326 = torch._C._nn.gelu(x_325)
        x_325 = None
        x_327 = torch._C._nn.linear(
            x_326,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_bias_,
        )
        x_326 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_bias_ = (None)
        x_328 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_parameters_layer_scale_parameter_
            * x_327
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_parameters_layer_scale_parameter_ = (
            x_327
        ) = None
        x_329 = x_328.permute(0, 3, 1, 2)
        x_328 = None
        x_330 = x_321 + x_329
        x_321 = x_329 = None
        x_331 = torch.conv2d(
            x_330,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            1536,
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_dwconv_parameters_bias_ = (None)
        x_332 = x_331.permute(0, 2, 3, 1)
        x_331 = None
        item_39 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_eps = (
            None
        )
        x_333 = torch.nn.functional.layer_norm(
            x_332,
            (1536,),
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_bias_,
            item_39,
        )
        x_332 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_layernorm_parameters_bias_ = (item_39) = (
            None
        )
        x_334 = torch._C._nn.linear(
            x_333,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_bias_,
        )
        x_333 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv1_parameters_bias_ = (None)
        x_335 = torch._C._nn.gelu(x_334)
        x_334 = None
        x_336 = torch._C._nn.linear(
            x_335,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_bias_,
        )
        x_335 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_modules_pwconv2_parameters_bias_ = (None)
        x_337 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_parameters_layer_scale_parameter_
            * x_336
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_2_parameters_layer_scale_parameter_ = (
            x_336
        ) = None
        x_338 = x_337.permute(0, 3, 1, 2)
        x_337 = None
        x_339 = x_330 + x_338
        x_330 = x_338 = None
        mean_8 = x_339.mean([-2, -1])
        item_40 = l_self_modules_layernorm_eps.item()
        l_self_modules_layernorm_eps = None
        pooled_output = torch.nn.functional.layer_norm(
            mean_8,
            (1536,),
            l_self_modules_layernorm_parameters_weight_,
            l_self_modules_layernorm_parameters_bias_,
            item_40,
        )
        mean_8 = (
            l_self_modules_layernorm_parameters_weight_
        ) = l_self_modules_layernorm_parameters_bias_ = item_40 = None
        return (x_339, pooled_output)
