import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s99: torch.SymInt,
        L_pixel_values_: torch.Tensor,
        L_self_modules_embedder_modules_convolution_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_embedder_modules_convolution_eps: torch.Tensor,
        L_self_modules_embedder_modules_pad_value: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_eps: torch.Tensor,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_eps: torch.Tensor,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_eps: torch.Tensor,
    ):
        l_pixel_values_ = L_pixel_values_
        l_self_modules_embedder_modules_convolution_parameters_weight_ = (
            L_self_modules_embedder_modules_convolution_parameters_weight_
        )
        l_self_modules_embedder_modules_convolution_eps = (
            L_self_modules_embedder_modules_convolution_eps
        )
        l_self_modules_embedder_modules_pad_value = (
            L_self_modules_embedder_modules_pad_value
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_eps
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_eps = L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_eps
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_eps = L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_eps
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_eps = L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_bias_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_bias_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_eps
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_ = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_eps = L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_eps
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_norm_eps = L_self_modules_norm_eps
        reshape = (
            l_self_modules_embedder_modules_convolution_parameters_weight_.reshape(
                1, 64, -1
            )
        )
        item = l_self_modules_embedder_modules_convolution_eps.item()
        l_self_modules_embedder_modules_convolution_eps = None
        batch_norm = torch.nn.functional.batch_norm(
            reshape, None, None, training=True, momentum=0.0, eps=item
        )
        reshape = item = None
        weight = batch_norm.reshape_as(
            l_self_modules_embedder_modules_convolution_parameters_weight_
        )
        batch_norm = (
            l_self_modules_embedder_modules_convolution_parameters_weight_
        ) = None
        hidden_state = torch.conv2d(
            l_pixel_values_, weight, None, (2, 2), (3, 3), (1, 1), 1
        )
        l_pixel_values_ = weight = None
        item_1 = l_self_modules_embedder_modules_pad_value.item()
        l_self_modules_embedder_modules_pad_value = None
        embedding = torch._C._nn.pad(hidden_state, (1, 1, 1, 1), "constant", item_1)
        hidden_state = item_1 = None
        embedding_1 = torch.nn.functional.max_pool2d(
            embedding, (3, 3), (2, 2), (0, 0), (1, 1), False
        )
        embedding = None
        item_2 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_eps = (
            None
        )
        hidden_state_1 = torch.nn.functional.group_norm(
            embedding_1,
            1,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_,
            item_2,
        )
        embedding_1 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_ = (item_2) = (
            None
        )
        hidden_state_2 = torch.nn.functional.relu(hidden_state_1, inplace=False)
        hidden_state_1 = None
        reshape_1 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 8, -1
        )
        item_3 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_eps = (
            None
        )
        batch_norm_1 = torch.nn.functional.batch_norm(
            reshape_1, None, None, training=True, momentum=0.0, eps=item_3
        )
        reshape_1 = item_3 = None
        weight_1 = batch_norm_1.reshape_as(
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_1 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        hidden_state_3 = torch.conv2d(
            hidden_state_2, weight_1, None, (1, 1), (0, 0), (1, 1), 1
        )
        weight_1 = None
        reshape_2 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 8, -1
        )
        item_4 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_eps = (
            None
        )
        batch_norm_2 = torch.nn.functional.batch_norm(
            reshape_2, None, None, training=True, momentum=0.0, eps=item_4
        )
        reshape_2 = item_4 = None
        weight_2 = batch_norm_2.reshape_as(
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_2 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_ = (None)
        hidden_state_4 = torch.conv2d(
            hidden_state_2, weight_2, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_2 = weight_2 = None
        item_5 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_eps = (
            None
        )
        hidden_state_5 = torch.nn.functional.group_norm(
            hidden_state_4,
            1,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_,
            item_5,
        )
        hidden_state_4 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_ = (item_5) = (
            None
        )
        hidden_state_6 = torch.nn.functional.relu(hidden_state_5, inplace=False)
        hidden_state_5 = None
        reshape_3 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 8, -1
        )
        item_6 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_eps = (
            None
        )
        batch_norm_3 = torch.nn.functional.batch_norm(
            reshape_3, None, None, training=True, momentum=0.0, eps=item_6
        )
        reshape_3 = item_6 = None
        weight_3 = batch_norm_3.reshape_as(
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_3 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_ = (None)
        hidden_state_7 = torch.conv2d(
            hidden_state_6, weight_3, None, (1, 1), (1, 1), (1, 1), 1
        )
        hidden_state_6 = weight_3 = None
        item_7 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_eps = (
            None
        )
        hidden_state_8 = torch.nn.functional.group_norm(
            hidden_state_7,
            1,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_bias_,
            item_7,
        )
        hidden_state_7 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_bias_ = (item_7) = (
            None
        )
        hidden_state_9 = torch.nn.functional.relu(hidden_state_8, inplace=False)
        hidden_state_8 = None
        reshape_4 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 8, -1
        )
        item_8 = (
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_eps = (
            None
        )
        batch_norm_4 = torch.nn.functional.batch_norm(
            reshape_4, None, None, training=True, momentum=0.0, eps=item_8
        )
        reshape_4 = item_8 = None
        weight_4 = batch_norm_4.reshape_as(
            l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_4 = l_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_ = (None)
        hidden_state_10 = torch.conv2d(
            hidden_state_9, weight_4, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_9 = weight_4 = None
        hidden_state_11 = hidden_state_10 + hidden_state_3
        hidden_state_10 = hidden_state_3 = None
        item_9 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_eps = (
            None
        )
        hidden_state_12 = torch.nn.functional.group_norm(
            hidden_state_11,
            1,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_,
            item_9,
        )
        hidden_state_11 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_ = (item_9) = (
            None
        )
        hidden_state_13 = torch.nn.functional.relu(hidden_state_12, inplace=False)
        hidden_state_12 = None
        reshape_5 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 16, -1
        )
        item_10 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_eps = (
            None
        )
        batch_norm_5 = torch.nn.functional.batch_norm(
            reshape_5, None, None, training=True, momentum=0.0, eps=item_10
        )
        reshape_5 = item_10 = None
        weight_5 = batch_norm_5.reshape_as(
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_5 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        hidden_state_14 = torch.conv2d(
            hidden_state_13, weight_5, None, (2, 2), (0, 0), (1, 1), 1
        )
        weight_5 = None
        reshape_6 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 8, -1
        )
        item_11 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_eps = (
            None
        )
        batch_norm_6 = torch.nn.functional.batch_norm(
            reshape_6, None, None, training=True, momentum=0.0, eps=item_11
        )
        reshape_6 = item_11 = None
        weight_6 = batch_norm_6.reshape_as(
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_6 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_ = (None)
        hidden_state_15 = torch.conv2d(
            hidden_state_13, weight_6, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_13 = weight_6 = None
        item_12 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_eps = (
            None
        )
        hidden_state_16 = torch.nn.functional.group_norm(
            hidden_state_15,
            1,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_,
            item_12,
        )
        hidden_state_15 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_ = (item_12) = (
            None
        )
        hidden_state_17 = torch.nn.functional.relu(hidden_state_16, inplace=False)
        hidden_state_16 = None
        reshape_7 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 8, -1
        )
        item_13 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_eps = (
            None
        )
        batch_norm_7 = torch.nn.functional.batch_norm(
            reshape_7, None, None, training=True, momentum=0.0, eps=item_13
        )
        reshape_7 = item_13 = None
        weight_7 = batch_norm_7.reshape_as(
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_7 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_ = (None)
        hidden_state_18 = torch.conv2d(
            hidden_state_17, weight_7, None, (2, 2), (1, 1), (1, 1), 1
        )
        hidden_state_17 = weight_7 = None
        item_14 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_eps = (
            None
        )
        hidden_state_19 = torch.nn.functional.group_norm(
            hidden_state_18,
            1,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_bias_,
            item_14,
        )
        hidden_state_18 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_bias_ = (item_14) = (
            None
        )
        hidden_state_20 = torch.nn.functional.relu(hidden_state_19, inplace=False)
        hidden_state_19 = None
        reshape_8 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 16, -1
        )
        item_15 = (
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_eps = (
            None
        )
        batch_norm_8 = torch.nn.functional.batch_norm(
            reshape_8, None, None, training=True, momentum=0.0, eps=item_15
        )
        reshape_8 = item_15 = None
        weight_8 = batch_norm_8.reshape_as(
            l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_8 = l_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_ = (None)
        hidden_state_21 = torch.conv2d(
            hidden_state_20, weight_8, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_20 = weight_8 = None
        hidden_state_22 = hidden_state_21 + hidden_state_14
        hidden_state_21 = hidden_state_14 = None
        item_16 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_eps = (
            None
        )
        hidden_state_23 = torch.nn.functional.group_norm(
            hidden_state_22,
            1,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_,
            item_16,
        )
        hidden_state_22 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_ = (item_16) = (
            None
        )
        hidden_state_24 = torch.nn.functional.relu(hidden_state_23, inplace=False)
        hidden_state_23 = None
        reshape_9 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 32, -1
        )
        item_17 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_eps = (
            None
        )
        batch_norm_9 = torch.nn.functional.batch_norm(
            reshape_9, None, None, training=True, momentum=0.0, eps=item_17
        )
        reshape_9 = item_17 = None
        weight_9 = batch_norm_9.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_9 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        hidden_state_25 = torch.conv2d(
            hidden_state_24, weight_9, None, (2, 2), (0, 0), (1, 1), 1
        )
        weight_9 = None
        reshape_10 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 8, -1
        )
        item_18 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_eps = (
            None
        )
        batch_norm_10 = torch.nn.functional.batch_norm(
            reshape_10, None, None, training=True, momentum=0.0, eps=item_18
        )
        reshape_10 = item_18 = None
        weight_10 = batch_norm_10.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_10 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_ = (None)
        hidden_state_26 = torch.conv2d(
            hidden_state_24, weight_10, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_24 = weight_10 = None
        item_19 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_eps = (
            None
        )
        hidden_state_27 = torch.nn.functional.group_norm(
            hidden_state_26,
            1,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_,
            item_19,
        )
        hidden_state_26 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_ = (item_19) = (
            None
        )
        hidden_state_28 = torch.nn.functional.relu(hidden_state_27, inplace=False)
        hidden_state_27 = None
        reshape_11 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 8, -1
        )
        item_20 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_eps = (
            None
        )
        batch_norm_11 = torch.nn.functional.batch_norm(
            reshape_11, None, None, training=True, momentum=0.0, eps=item_20
        )
        reshape_11 = item_20 = None
        weight_11 = batch_norm_11.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_11 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_ = (None)
        hidden_state_29 = torch.conv2d(
            hidden_state_28, weight_11, None, (2, 2), (1, 1), (1, 1), 1
        )
        hidden_state_28 = weight_11 = None
        item_21 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_eps = (
            None
        )
        hidden_state_30 = torch.nn.functional.group_norm(
            hidden_state_29,
            1,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_bias_,
            item_21,
        )
        hidden_state_29 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_bias_ = (item_21) = (
            None
        )
        hidden_state_31 = torch.nn.functional.relu(hidden_state_30, inplace=False)
        hidden_state_30 = None
        reshape_12 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 32, -1
        )
        item_22 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_eps = (
            None
        )
        batch_norm_12 = torch.nn.functional.batch_norm(
            reshape_12, None, None, training=True, momentum=0.0, eps=item_22
        )
        reshape_12 = item_22 = None
        weight_12 = batch_norm_12.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_12 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_ = (None)
        hidden_state_32 = torch.conv2d(
            hidden_state_31, weight_12, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_31 = weight_12 = None
        hidden_state_33 = hidden_state_32 + hidden_state_25
        hidden_state_32 = hidden_state_25 = None
        item_23 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_eps = (
            None
        )
        hidden_state_34 = torch.nn.functional.group_norm(
            hidden_state_33,
            1,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_,
            item_23,
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_ = (item_23) = (
            None
        )
        hidden_state_35 = torch.nn.functional.relu(hidden_state_34, inplace=False)
        hidden_state_34 = None
        reshape_13 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_.reshape(
            1, 8, -1
        )
        item_24 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_eps = (
            None
        )
        batch_norm_13 = torch.nn.functional.batch_norm(
            reshape_13, None, None, training=True, momentum=0.0, eps=item_24
        )
        reshape_13 = item_24 = None
        weight_13 = batch_norm_13.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_
        )
        batch_norm_13 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_ = (None)
        hidden_state_36 = torch.conv2d(
            hidden_state_35, weight_13, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_35 = weight_13 = None
        item_25 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_eps = (
            None
        )
        hidden_state_37 = torch.nn.functional.group_norm(
            hidden_state_36,
            1,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_,
            item_25,
        )
        hidden_state_36 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_ = (item_25) = (
            None
        )
        hidden_state_38 = torch.nn.functional.relu(hidden_state_37, inplace=False)
        hidden_state_37 = None
        reshape_14 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_.reshape(
            1, 8, -1
        )
        item_26 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_eps = (
            None
        )
        batch_norm_14 = torch.nn.functional.batch_norm(
            reshape_14, None, None, training=True, momentum=0.0, eps=item_26
        )
        reshape_14 = item_26 = None
        weight_14 = batch_norm_14.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_
        )
        batch_norm_14 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_ = (None)
        hidden_state_39 = torch.conv2d(
            hidden_state_38, weight_14, None, (1, 1), (1, 1), (1, 1), 1
        )
        hidden_state_38 = weight_14 = None
        item_27 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_eps = (
            None
        )
        hidden_state_40 = torch.nn.functional.group_norm(
            hidden_state_39,
            1,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_bias_,
            item_27,
        )
        hidden_state_39 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_bias_ = (item_27) = (
            None
        )
        hidden_state_41 = torch.nn.functional.relu(hidden_state_40, inplace=False)
        hidden_state_40 = None
        reshape_15 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_.reshape(
            1, 32, -1
        )
        item_28 = (
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_eps = (
            None
        )
        batch_norm_15 = torch.nn.functional.batch_norm(
            reshape_15, None, None, training=True, momentum=0.0, eps=item_28
        )
        reshape_15 = item_28 = None
        weight_15 = batch_norm_15.reshape_as(
            l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_
        )
        batch_norm_15 = l_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_ = (None)
        hidden_state_42 = torch.conv2d(
            hidden_state_41, weight_15, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_41 = weight_15 = None
        hidden_state_43 = hidden_state_42 + hidden_state_33
        hidden_state_42 = hidden_state_33 = None
        item_29 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_eps = (
            None
        )
        hidden_state_44 = torch.nn.functional.group_norm(
            hidden_state_43,
            1,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_,
            item_29,
        )
        hidden_state_43 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_ = (item_29) = (
            None
        )
        hidden_state_45 = torch.nn.functional.relu(hidden_state_44, inplace=False)
        hidden_state_44 = None
        reshape_16 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_.reshape(
            1, 64, -1
        )
        item_30 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_eps = (
            None
        )
        batch_norm_16 = torch.nn.functional.batch_norm(
            reshape_16, None, None, training=True, momentum=0.0, eps=item_30
        )
        reshape_16 = item_30 = None
        weight_16 = batch_norm_16.reshape_as(
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        )
        batch_norm_16 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        hidden_state_46 = torch.conv2d(
            hidden_state_45, weight_16, None, (2, 2), (0, 0), (1, 1), 1
        )
        weight_16 = None
        reshape_17 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_.reshape(
            1, 16, -1
        )
        item_31 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_eps = (
            None
        )
        batch_norm_17 = torch.nn.functional.batch_norm(
            reshape_17, None, None, training=True, momentum=0.0, eps=item_31
        )
        reshape_17 = item_31 = None
        weight_17 = batch_norm_17.reshape_as(
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_
        )
        batch_norm_17 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_ = (None)
        hidden_state_47 = torch.conv2d(
            hidden_state_45, weight_17, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_45 = weight_17 = None
        item_32 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_eps = (
            None
        )
        hidden_state_48 = torch.nn.functional.group_norm(
            hidden_state_47,
            1,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_,
            item_32,
        )
        hidden_state_47 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_ = (item_32) = (
            None
        )
        hidden_state_49 = torch.nn.functional.relu(hidden_state_48, inplace=False)
        hidden_state_48 = None
        reshape_18 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_.reshape(
            1, 16, -1
        )
        item_33 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_eps = (
            None
        )
        batch_norm_18 = torch.nn.functional.batch_norm(
            reshape_18, None, None, training=True, momentum=0.0, eps=item_33
        )
        reshape_18 = item_33 = None
        weight_18 = batch_norm_18.reshape_as(
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_
        )
        batch_norm_18 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_ = (None)
        hidden_state_50 = torch.conv2d(
            hidden_state_49, weight_18, None, (2, 2), (1, 1), (1, 1), 1
        )
        hidden_state_49 = weight_18 = None
        item_34 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_eps = (
            None
        )
        hidden_state_51 = torch.nn.functional.group_norm(
            hidden_state_50,
            1,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_bias_,
            item_34,
        )
        hidden_state_50 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_weight_ = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_bias_ = (item_34) = (
            None
        )
        hidden_state_52 = torch.nn.functional.relu(hidden_state_51, inplace=False)
        hidden_state_51 = None
        reshape_19 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_.reshape(
            1, 64, -1
        )
        item_35 = (
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_eps.item()
        )
        l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_eps = (
            None
        )
        batch_norm_19 = torch.nn.functional.batch_norm(
            reshape_19, None, None, training=True, momentum=0.0, eps=item_35
        )
        reshape_19 = item_35 = None
        weight_19 = batch_norm_19.reshape_as(
            l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_
        )
        batch_norm_19 = l_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_ = (None)
        hidden_state_53 = torch.conv2d(
            hidden_state_52, weight_19, None, (1, 1), (0, 0), (1, 1), 1
        )
        hidden_state_52 = weight_19 = None
        hidden_state_54 = hidden_state_53 + hidden_state_46
        hidden_state_53 = hidden_state_46 = None
        item_36 = l_self_modules_norm_eps.item()
        l_self_modules_norm_eps = None
        hidden_state_55 = torch.nn.functional.group_norm(
            hidden_state_54,
            1,
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            item_36,
        )
        hidden_state_54 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = item_36 = None
        hidden_state_56 = torch.nn.functional.relu(hidden_state_55, inplace=False)
        hidden_state_55 = None
        pooled_output = torch.nn.functional.adaptive_avg_pool2d(hidden_state_56, (1, 1))
        return (hidden_state_56, pooled_output)
