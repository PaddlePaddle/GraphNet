import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_feature_maps_0_: torch.Tensor,
        L_self_modules_neck_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_1_: torch.Tensor,
        L_self_modules_neck_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_2_: torch.Tensor,
        L_self_modules_neck_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        s62: torch.SymInt,
        s43: torch.SymInt,
        L_stack0_feature_maps_3_: torch.Tensor,
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
        L_self_modules_head_modules_head_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_1_scale_factor: torch.Tensor,
        L_self_modules_head_modules_head_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_feature_maps_0_ = L_stack0_feature_maps_0_
        l_self_modules_neck_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_0_parameters_weight_
        )
        l_stack0_feature_maps_1_ = L_stack0_feature_maps_1_
        l_self_modules_neck_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_1_parameters_weight_
        )
        l_stack0_feature_maps_2_ = L_stack0_feature_maps_2_
        l_self_modules_neck_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_neck_modules_convs_modules_2_parameters_weight_
        )
        l_stack0_feature_maps_3_ = L_stack0_feature_maps_3_
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
        l_self_modules_head_modules_head_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_head_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_head_modules_0_parameters_bias_ = (
            L_self_modules_head_modules_head_modules_0_parameters_bias_
        )
        l_self_modules_head_modules_head_modules_1_scale_factor = (
            L_self_modules_head_modules_head_modules_1_scale_factor
        )
        l_self_modules_head_modules_head_modules_2_parameters_weight_ = (
            L_self_modules_head_modules_head_modules_2_parameters_weight_
        )
        l_self_modules_head_modules_head_modules_2_parameters_bias_ = (
            L_self_modules_head_modules_head_modules_2_parameters_bias_
        )
        l_self_modules_head_modules_head_modules_4_parameters_weight_ = (
            L_self_modules_head_modules_head_modules_4_parameters_weight_
        )
        l_self_modules_head_modules_head_modules_4_parameters_bias_ = (
            L_self_modules_head_modules_head_modules_4_parameters_bias_
        )
        hidden_state_34 = torch.conv2d(
            l_stack0_feature_maps_0_,
            l_self_modules_neck_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_0_ = (
            l_self_modules_neck_modules_convs_modules_0_parameters_weight_
        ) = None
        hidden_state_21 = torch.conv2d(
            l_stack0_feature_maps_1_,
            l_self_modules_neck_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_1_ = (
            l_self_modules_neck_modules_convs_modules_1_parameters_weight_
        ) = None
        hidden_state_8 = torch.conv2d(
            l_stack0_feature_maps_2_,
            l_self_modules_neck_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_2_ = (
            l_self_modules_neck_modules_convs_modules_2_parameters_weight_
        ) = None
        hidden_state = torch.conv2d(
            l_stack0_feature_maps_3_,
            l_self_modules_neck_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_3_ = (
            l_self_modules_neck_modules_convs_modules_3_parameters_weight_
        ) = None
        hidden_state_1 = torch.nn.functional.relu(hidden_state, inplace=False)
        hidden_state_2 = torch.conv2d(
            hidden_state_1,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_1 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_3 = torch.nn.functional.relu(hidden_state_2, inplace=False)
        hidden_state_2 = None
        hidden_state_4 = torch.conv2d(
            hidden_state_3,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_3 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_5 = hidden_state_4 + hidden_state
        hidden_state_4 = hidden_state = None
        hidden_state_6 = torch.nn.functional.interpolate(
            hidden_state_5, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_5 = None
        hidden_state_7 = torch.conv2d(
            hidden_state_6,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_6 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = (None)
        hidden_state_9 = torch.nn.functional.relu(hidden_state_8, inplace=False)
        hidden_state_10 = torch.conv2d(
            hidden_state_9,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_9 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_11 = torch.nn.functional.relu(hidden_state_10, inplace=False)
        hidden_state_10 = None
        hidden_state_12 = torch.conv2d(
            hidden_state_11,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_11 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_1 = hidden_state_12 + hidden_state_8
        hidden_state_12 = hidden_state_8 = None
        hidden_state_13 = hidden_state_7 + add_1
        hidden_state_7 = add_1 = None
        hidden_state_14 = torch.nn.functional.relu(hidden_state_13, inplace=False)
        hidden_state_15 = torch.conv2d(
            hidden_state_14,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_14 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_16 = torch.nn.functional.relu(hidden_state_15, inplace=False)
        hidden_state_15 = None
        hidden_state_17 = torch.conv2d(
            hidden_state_16,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_16 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_18 = hidden_state_17 + hidden_state_13
        hidden_state_17 = hidden_state_13 = None
        hidden_state_19 = torch.nn.functional.interpolate(
            hidden_state_18, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_18 = None
        hidden_state_20 = torch.conv2d(
            hidden_state_19,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_19 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = (None)
        hidden_state_22 = torch.nn.functional.relu(hidden_state_21, inplace=False)
        hidden_state_23 = torch.conv2d(
            hidden_state_22,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_22 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_24 = torch.nn.functional.relu(hidden_state_23, inplace=False)
        hidden_state_23 = None
        hidden_state_25 = torch.conv2d(
            hidden_state_24,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_24 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_4 = hidden_state_25 + hidden_state_21
        hidden_state_25 = hidden_state_21 = None
        hidden_state_26 = hidden_state_20 + add_4
        hidden_state_20 = add_4 = None
        hidden_state_27 = torch.nn.functional.relu(hidden_state_26, inplace=False)
        hidden_state_28 = torch.conv2d(
            hidden_state_27,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_27 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_29 = torch.nn.functional.relu(hidden_state_28, inplace=False)
        hidden_state_28 = None
        hidden_state_30 = torch.conv2d(
            hidden_state_29,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_29 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_31 = hidden_state_30 + hidden_state_26
        hidden_state_30 = hidden_state_26 = None
        hidden_state_32 = torch.nn.functional.interpolate(
            hidden_state_31, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_31 = None
        hidden_state_33 = torch.conv2d(
            hidden_state_32,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_32 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = (None)
        hidden_state_35 = torch.nn.functional.relu(hidden_state_34, inplace=False)
        hidden_state_36 = torch.conv2d(
            hidden_state_35,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_35 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_37 = torch.nn.functional.relu(hidden_state_36, inplace=False)
        hidden_state_36 = None
        hidden_state_38 = torch.conv2d(
            hidden_state_37,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_37 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_7 = hidden_state_38 + hidden_state_34
        hidden_state_38 = hidden_state_34 = None
        hidden_state_39 = hidden_state_33 + add_7
        hidden_state_33 = add_7 = None
        hidden_state_40 = torch.nn.functional.relu(hidden_state_39, inplace=False)
        hidden_state_41 = torch.conv2d(
            hidden_state_40,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_40 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_42 = torch.nn.functional.relu(hidden_state_41, inplace=False)
        hidden_state_41 = None
        hidden_state_43 = torch.conv2d(
            hidden_state_42,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_42 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_44 = hidden_state_43 + hidden_state_39
        hidden_state_43 = hidden_state_39 = None
        hidden_state_45 = torch.nn.functional.interpolate(
            hidden_state_44, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_44 = None
        hidden_state_46 = torch.conv2d(
            hidden_state_45,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_45 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = (None)
        input_1 = torch.conv2d(
            hidden_state_46,
            l_self_modules_head_modules_head_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_46 = (
            l_self_modules_head_modules_head_modules_0_parameters_weight_
        ) = l_self_modules_head_modules_head_modules_0_parameters_bias_ = None
        item = l_self_modules_head_modules_head_modules_1_scale_factor.item()
        l_self_modules_head_modules_head_modules_1_scale_factor = None
        input_2 = torch.nn.functional.interpolate(
            input_1, None, item, "bilinear", True, recompute_scale_factor=None
        )
        input_1 = item = None
        input_3 = torch.conv2d(
            input_2,
            l_self_modules_head_modules_head_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_modules_2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = (
            l_self_modules_head_modules_head_modules_2_parameters_weight_
        ) = l_self_modules_head_modules_head_modules_2_parameters_bias_ = None
        input_4 = torch.nn.functional.relu(input_3, inplace=False)
        input_3 = None
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_head_modules_head_modules_4_parameters_weight_,
            l_self_modules_head_modules_head_modules_4_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_4 = (
            l_self_modules_head_modules_head_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_head_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=False)
        input_5 = None
        predicted_depth = input_6.squeeze(dim=1)
        input_6 = None
        return (predicted_depth,)
