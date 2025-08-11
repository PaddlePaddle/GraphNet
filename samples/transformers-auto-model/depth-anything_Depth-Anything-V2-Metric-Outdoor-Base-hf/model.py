import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_feature_maps_0_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_1_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_2_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_3_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_conv3_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_feature_maps_0_ = L_stack0_feature_maps_0_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_
        l_stack0_feature_maps_1_ = L_stack0_feature_maps_1_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_
        l_stack0_feature_maps_2_ = L_stack0_feature_maps_2_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_
        l_stack0_feature_maps_3_ = L_stack0_feature_maps_3_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_
        l_self_modules_neck_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_neck_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_neck_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_neck_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_
        l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_
        l_self_modules_head_modules_conv1_parameters_weight_ = (
            L_self_modules_head_modules_conv1_parameters_weight_
        )
        l_self_modules_head_modules_conv1_parameters_bias_ = (
            L_self_modules_head_modules_conv1_parameters_bias_
        )
        l_self_modules_head_modules_conv2_parameters_weight_ = (
            L_self_modules_head_modules_conv2_parameters_weight_
        )
        l_self_modules_head_modules_conv2_parameters_bias_ = (
            L_self_modules_head_modules_conv2_parameters_bias_
        )
        l_self_modules_head_modules_conv3_parameters_weight_ = (
            L_self_modules_head_modules_conv3_parameters_weight_
        )
        l_self_modules_head_modules_conv3_parameters_bias_ = (
            L_self_modules_head_modules_conv3_parameters_bias_
        )
        hidden_state = l_stack0_feature_maps_0_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_0_ = None
        hidden_state_1 = hidden_state.reshape(1, 37, 37, 768)
        hidden_state = None
        permute = hidden_state_1.permute(0, 3, 1, 2)
        hidden_state_1 = None
        hidden_state_2 = permute.contiguous()
        permute = None
        hidden_state_3 = torch.conv2d(
            hidden_state_2,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_2 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = (None)
        hidden_state_4 = torch.conv_transpose2d(
            hidden_state_3,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_,
            (4, 4),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        hidden_state_3 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_ = (None)
        hidden_state_5 = l_stack0_feature_maps_1_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_1_ = None
        hidden_state_6 = hidden_state_5.reshape(1, 37, 37, 768)
        hidden_state_5 = None
        permute_1 = hidden_state_6.permute(0, 3, 1, 2)
        hidden_state_6 = None
        hidden_state_7 = permute_1.contiguous()
        permute_1 = None
        hidden_state_8 = torch.conv2d(
            hidden_state_7,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_7 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = (None)
        hidden_state_9 = torch.conv_transpose2d(
            hidden_state_8,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        hidden_state_8 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_ = (None)
        hidden_state_10 = l_stack0_feature_maps_2_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_2_ = None
        hidden_state_11 = hidden_state_10.reshape(1, 37, 37, 768)
        hidden_state_10 = None
        permute_2 = hidden_state_11.permute(0, 3, 1, 2)
        hidden_state_11 = None
        hidden_state_12 = permute_2.contiguous()
        permute_2 = None
        hidden_state_13 = torch.conv2d(
            hidden_state_12,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_12 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = (None)
        hidden_state_14 = l_stack0_feature_maps_3_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_3_ = None
        hidden_state_15 = hidden_state_14.reshape(1, 37, 37, 768)
        hidden_state_14 = None
        permute_3 = hidden_state_15.permute(0, 3, 1, 2)
        hidden_state_15 = None
        hidden_state_16 = permute_3.contiguous()
        permute_3 = None
        hidden_state_17 = torch.conv2d(
            hidden_state_16,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_16 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = (None)
        hidden_state_18 = torch.conv2d(
            hidden_state_17,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_17 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_ = (None)
        hidden_state_53 = torch.conv2d(
            hidden_state_4,
            l_self_modules_neck_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_4 = (
            l_self_modules_neck_modules_convs_modules_0_parameters_weight_
        ) = None
        hidden_state_40 = torch.conv2d(
            hidden_state_9,
            l_self_modules_neck_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_9 = (
            l_self_modules_neck_modules_convs_modules_1_parameters_weight_
        ) = None
        hidden_state_27 = torch.conv2d(
            hidden_state_13,
            l_self_modules_neck_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_13 = (
            l_self_modules_neck_modules_convs_modules_2_parameters_weight_
        ) = None
        hidden_state_19 = torch.conv2d(
            hidden_state_18,
            l_self_modules_neck_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_18 = (
            l_self_modules_neck_modules_convs_modules_3_parameters_weight_
        ) = None
        hidden_state_20 = torch.nn.functional.relu(hidden_state_19, inplace=False)
        hidden_state_21 = torch.conv2d(
            hidden_state_20,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_20 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_22 = torch.nn.functional.relu(hidden_state_21, inplace=False)
        hidden_state_21 = None
        hidden_state_23 = torch.conv2d(
            hidden_state_22,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_22 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_24 = hidden_state_23 + hidden_state_19
        hidden_state_23 = hidden_state_19 = None
        hidden_state_25 = torch.nn.functional.interpolate(
            hidden_state_24, size=(37, 37), mode="bilinear", align_corners=True
        )
        hidden_state_24 = None
        hidden_state_26 = torch.conv2d(
            hidden_state_25,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_25 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = (None)
        hidden_state_28 = torch.nn.functional.relu(hidden_state_27, inplace=False)
        hidden_state_29 = torch.conv2d(
            hidden_state_28,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_28 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_30 = torch.nn.functional.relu(hidden_state_29, inplace=False)
        hidden_state_29 = None
        hidden_state_31 = torch.conv2d(
            hidden_state_30,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_30 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_1 = hidden_state_31 + hidden_state_27
        hidden_state_31 = hidden_state_27 = None
        hidden_state_32 = hidden_state_26 + add_1
        hidden_state_26 = add_1 = None
        hidden_state_33 = torch.nn.functional.relu(hidden_state_32, inplace=False)
        hidden_state_34 = torch.conv2d(
            hidden_state_33,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_33 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_35 = torch.nn.functional.relu(hidden_state_34, inplace=False)
        hidden_state_34 = None
        hidden_state_36 = torch.conv2d(
            hidden_state_35,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_35 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_37 = hidden_state_36 + hidden_state_32
        hidden_state_36 = hidden_state_32 = None
        hidden_state_38 = torch.nn.functional.interpolate(
            hidden_state_37, size=(74, 74), mode="bilinear", align_corners=True
        )
        hidden_state_37 = None
        hidden_state_39 = torch.conv2d(
            hidden_state_38,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_38 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = (None)
        hidden_state_41 = torch.nn.functional.relu(hidden_state_40, inplace=False)
        hidden_state_42 = torch.conv2d(
            hidden_state_41,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_41 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_43 = torch.nn.functional.relu(hidden_state_42, inplace=False)
        hidden_state_42 = None
        hidden_state_44 = torch.conv2d(
            hidden_state_43,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_43 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_4 = hidden_state_44 + hidden_state_40
        hidden_state_44 = hidden_state_40 = None
        hidden_state_45 = hidden_state_39 + add_4
        hidden_state_39 = add_4 = None
        hidden_state_46 = torch.nn.functional.relu(hidden_state_45, inplace=False)
        hidden_state_47 = torch.conv2d(
            hidden_state_46,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_46 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_48 = torch.nn.functional.relu(hidden_state_47, inplace=False)
        hidden_state_47 = None
        hidden_state_49 = torch.conv2d(
            hidden_state_48,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_48 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_50 = hidden_state_49 + hidden_state_45
        hidden_state_49 = hidden_state_45 = None
        hidden_state_51 = torch.nn.functional.interpolate(
            hidden_state_50, size=(148, 148), mode="bilinear", align_corners=True
        )
        hidden_state_50 = None
        hidden_state_52 = torch.conv2d(
            hidden_state_51,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_51 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = (None)
        hidden_state_54 = torch.nn.functional.relu(hidden_state_53, inplace=False)
        hidden_state_55 = torch.conv2d(
            hidden_state_54,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_54 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_56 = torch.nn.functional.relu(hidden_state_55, inplace=False)
        hidden_state_55 = None
        hidden_state_57 = torch.conv2d(
            hidden_state_56,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_56 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_7 = hidden_state_57 + hidden_state_53
        hidden_state_57 = hidden_state_53 = None
        hidden_state_58 = hidden_state_52 + add_7
        hidden_state_52 = add_7 = None
        hidden_state_59 = torch.nn.functional.relu(hidden_state_58, inplace=False)
        hidden_state_60 = torch.conv2d(
            hidden_state_59,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_59 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_61 = torch.nn.functional.relu(hidden_state_60, inplace=False)
        hidden_state_60 = None
        hidden_state_62 = torch.conv2d(
            hidden_state_61,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_61 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_63 = hidden_state_62 + hidden_state_58
        hidden_state_62 = hidden_state_58 = None
        hidden_state_64 = torch.nn.functional.interpolate(
            hidden_state_63, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_63 = None
        hidden_state_65 = torch.conv2d(
            hidden_state_64,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_64 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = (None)
        predicted_depth = torch.conv2d(
            hidden_state_65,
            l_self_modules_head_modules_conv1_parameters_weight_,
            l_self_modules_head_modules_conv1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_65 = (
            l_self_modules_head_modules_conv1_parameters_weight_
        ) = l_self_modules_head_modules_conv1_parameters_bias_ = None
        predicted_depth_1 = torch.nn.functional.interpolate(
            predicted_depth, (518, 518), mode="bilinear", align_corners=True
        )
        predicted_depth = None
        predicted_depth_2 = torch.conv2d(
            predicted_depth_1,
            l_self_modules_head_modules_conv2_parameters_weight_,
            l_self_modules_head_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        predicted_depth_1 = (
            l_self_modules_head_modules_conv2_parameters_weight_
        ) = l_self_modules_head_modules_conv2_parameters_bias_ = None
        predicted_depth_3 = torch.nn.functional.relu(predicted_depth_2, inplace=False)
        predicted_depth_2 = None
        predicted_depth_4 = torch.conv2d(
            predicted_depth_3,
            l_self_modules_head_modules_conv3_parameters_weight_,
            l_self_modules_head_modules_conv3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        predicted_depth_3 = (
            l_self_modules_head_modules_conv3_parameters_weight_
        ) = l_self_modules_head_modules_conv3_parameters_bias_ = None
        sigmoid = torch.sigmoid(predicted_depth_4)
        predicted_depth_4 = None
        predicted_depth_5 = sigmoid * 80
        sigmoid = None
        predicted_depth_6 = predicted_depth_5.squeeze(dim=1)
        predicted_depth_5 = None
        return (predicted_depth_6,)
