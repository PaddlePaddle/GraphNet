import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_stack0_feature_maps_0_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_1_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_2_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_3_: torch.Tensor,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_head_modules_head_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_head_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_stack0_feature_maps_0_ = L_stack0_feature_maps_0_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_
        l_stack0_feature_maps_1_ = L_stack0_feature_maps_1_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_
        l_stack0_feature_maps_2_ = L_stack0_feature_maps_2_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_
        l_stack0_feature_maps_3_ = L_stack0_feature_maps_3_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_
        l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_ = L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_
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
        l_self_modules_head_modules_head_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_head_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_head_modules_0_parameters_bias_ = (
            L_self_modules_head_modules_head_modules_0_parameters_bias_
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
        cls_token = l_stack0_feature_maps_0_[(slice(None, None, None), 0)]
        hidden_state = l_stack0_feature_maps_0_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_0_ = None
        hidden_state_1 = hidden_state.reshape(1, 24, 24, 768)
        hidden_state = None
        permute = hidden_state_1.permute(0, 3, 1, 2)
        hidden_state_1 = None
        hidden_state_2 = permute.contiguous()
        permute = None
        flatten = hidden_state_2.flatten(2)
        hidden_state_2 = None
        hidden_state_3 = flatten.permute((0, 2, 1))
        flatten = None
        unsqueeze = cls_token.unsqueeze(1)
        cls_token = None
        readout = unsqueeze.expand_as(hidden_state_3)
        unsqueeze = None
        cat = torch.cat((hidden_state_3, readout), -1)
        hidden_state_3 = readout = None
        input_1 = torch._C._nn.linear(
            cat,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_,
        )
        cat = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_ = (None)
        input_2 = torch._C._nn.gelu(input_1)
        input_1 = None
        permute_2 = input_2.permute(0, 2, 1)
        input_2 = None
        hidden_state_4 = permute_2.reshape((1, 768, 24, 24))
        permute_2 = None
        hidden_state_5 = torch.conv2d(
            hidden_state_4,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_4 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = (None)
        hidden_state_6 = torch.conv_transpose2d(
            hidden_state_5,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_,
            (4, 4),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        hidden_state_5 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_ = (None)
        cls_token_1 = l_stack0_feature_maps_1_[(slice(None, None, None), 0)]
        hidden_state_7 = l_stack0_feature_maps_1_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_1_ = None
        hidden_state_8 = hidden_state_7.reshape(1, 24, 24, 768)
        hidden_state_7 = None
        permute_3 = hidden_state_8.permute(0, 3, 1, 2)
        hidden_state_8 = None
        hidden_state_9 = permute_3.contiguous()
        permute_3 = None
        flatten_1 = hidden_state_9.flatten(2)
        hidden_state_9 = None
        hidden_state_10 = flatten_1.permute((0, 2, 1))
        flatten_1 = None
        unsqueeze_1 = cls_token_1.unsqueeze(1)
        cls_token_1 = None
        readout_1 = unsqueeze_1.expand_as(hidden_state_10)
        unsqueeze_1 = None
        cat_1 = torch.cat((hidden_state_10, readout_1), -1)
        hidden_state_10 = readout_1 = None
        input_3 = torch._C._nn.linear(
            cat_1,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_,
        )
        cat_1 = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_ = (None)
        input_4 = torch._C._nn.gelu(input_3)
        input_3 = None
        permute_5 = input_4.permute(0, 2, 1)
        input_4 = None
        hidden_state_11 = permute_5.reshape((1, 768, 24, 24))
        permute_5 = None
        hidden_state_12 = torch.conv2d(
            hidden_state_11,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_11 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = (None)
        hidden_state_13 = torch.conv_transpose2d(
            hidden_state_12,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_,
            (2, 2),
            (0, 0),
            (0, 0),
            1,
            (1, 1),
        )
        hidden_state_12 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_ = (None)
        cls_token_2 = l_stack0_feature_maps_2_[(slice(None, None, None), 0)]
        hidden_state_14 = l_stack0_feature_maps_2_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_2_ = None
        hidden_state_15 = hidden_state_14.reshape(1, 24, 24, 768)
        hidden_state_14 = None
        permute_6 = hidden_state_15.permute(0, 3, 1, 2)
        hidden_state_15 = None
        hidden_state_16 = permute_6.contiguous()
        permute_6 = None
        flatten_2 = hidden_state_16.flatten(2)
        hidden_state_16 = None
        hidden_state_17 = flatten_2.permute((0, 2, 1))
        flatten_2 = None
        unsqueeze_2 = cls_token_2.unsqueeze(1)
        cls_token_2 = None
        readout_2 = unsqueeze_2.expand_as(hidden_state_17)
        unsqueeze_2 = None
        cat_2 = torch.cat((hidden_state_17, readout_2), -1)
        hidden_state_17 = readout_2 = None
        input_5 = torch._C._nn.linear(
            cat_2,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_,
        )
        cat_2 = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_ = (None)
        input_6 = torch._C._nn.gelu(input_5)
        input_5 = None
        permute_8 = input_6.permute(0, 2, 1)
        input_6 = None
        hidden_state_18 = permute_8.reshape((1, 768, 24, 24))
        permute_8 = None
        hidden_state_19 = torch.conv2d(
            hidden_state_18,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_18 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = (None)
        cls_token_3 = l_stack0_feature_maps_3_[(slice(None, None, None), 0)]
        hidden_state_20 = l_stack0_feature_maps_3_[
            (slice(None, None, None), slice(1, None, None))
        ]
        l_stack0_feature_maps_3_ = None
        hidden_state_21 = hidden_state_20.reshape(1, 24, 24, 768)
        hidden_state_20 = None
        permute_9 = hidden_state_21.permute(0, 3, 1, 2)
        hidden_state_21 = None
        hidden_state_22 = permute_9.contiguous()
        permute_9 = None
        flatten_3 = hidden_state_22.flatten(2)
        hidden_state_22 = None
        hidden_state_23 = flatten_3.permute((0, 2, 1))
        flatten_3 = None
        unsqueeze_3 = cls_token_3.unsqueeze(1)
        cls_token_3 = None
        readout_3 = unsqueeze_3.expand_as(hidden_state_23)
        unsqueeze_3 = None
        cat_3 = torch.cat((hidden_state_23, readout_3), -1)
        hidden_state_23 = readout_3 = None
        input_7 = torch._C._nn.linear(
            cat_3,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_,
        )
        cat_3 = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_ = (None)
        input_8 = torch._C._nn.gelu(input_7)
        input_7 = None
        permute_11 = input_8.permute(0, 2, 1)
        input_8 = None
        hidden_state_24 = permute_11.reshape((1, 768, 24, 24))
        permute_11 = None
        hidden_state_25 = torch.conv2d(
            hidden_state_24,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_24 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = (None)
        hidden_state_26 = torch.conv2d(
            hidden_state_25,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_,
            l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_25 = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_ = l_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_ = (None)
        hidden_state_61 = torch.conv2d(
            hidden_state_6,
            l_self_modules_neck_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_6 = (
            l_self_modules_neck_modules_convs_modules_0_parameters_weight_
        ) = None
        hidden_state_48 = torch.conv2d(
            hidden_state_13,
            l_self_modules_neck_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_13 = (
            l_self_modules_neck_modules_convs_modules_1_parameters_weight_
        ) = None
        hidden_state_35 = torch.conv2d(
            hidden_state_19,
            l_self_modules_neck_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_19 = (
            l_self_modules_neck_modules_convs_modules_2_parameters_weight_
        ) = None
        hidden_state_27 = torch.conv2d(
            hidden_state_26,
            l_self_modules_neck_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_26 = (
            l_self_modules_neck_modules_convs_modules_3_parameters_weight_
        ) = None
        hidden_state_28 = torch.nn.functional.relu(hidden_state_27, inplace=False)
        hidden_state_29 = torch.conv2d(
            hidden_state_28,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_28 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_30 = torch.nn.functional.relu(hidden_state_29, inplace=False)
        hidden_state_29 = None
        hidden_state_31 = torch.conv2d(
            hidden_state_30,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_30 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_32 = hidden_state_31 + hidden_state_27
        hidden_state_31 = hidden_state_27 = None
        hidden_state_33 = torch.nn.functional.interpolate(
            hidden_state_32, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_32 = None
        hidden_state_34 = torch.conv2d(
            hidden_state_33,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_33 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_ = (None)
        hidden_state_36 = torch.nn.functional.relu(hidden_state_35, inplace=False)
        hidden_state_37 = torch.conv2d(
            hidden_state_36,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_36 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_38 = torch.nn.functional.relu(hidden_state_37, inplace=False)
        hidden_state_37 = None
        hidden_state_39 = torch.conv2d(
            hidden_state_38,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_38 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_1 = hidden_state_39 + hidden_state_35
        hidden_state_39 = hidden_state_35 = None
        hidden_state_40 = hidden_state_34 + add_1
        hidden_state_34 = add_1 = None
        hidden_state_41 = torch.nn.functional.relu(hidden_state_40, inplace=False)
        hidden_state_42 = torch.conv2d(
            hidden_state_41,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_41 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_43 = torch.nn.functional.relu(hidden_state_42, inplace=False)
        hidden_state_42 = None
        hidden_state_44 = torch.conv2d(
            hidden_state_43,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_43 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_45 = hidden_state_44 + hidden_state_40
        hidden_state_44 = hidden_state_40 = None
        hidden_state_46 = torch.nn.functional.interpolate(
            hidden_state_45, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_45 = None
        hidden_state_47 = torch.conv2d(
            hidden_state_46,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_46 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_ = (None)
        hidden_state_49 = torch.nn.functional.relu(hidden_state_48, inplace=False)
        hidden_state_50 = torch.conv2d(
            hidden_state_49,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_49 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_51 = torch.nn.functional.relu(hidden_state_50, inplace=False)
        hidden_state_50 = None
        hidden_state_52 = torch.conv2d(
            hidden_state_51,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_51 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_4 = hidden_state_52 + hidden_state_48
        hidden_state_52 = hidden_state_48 = None
        hidden_state_53 = hidden_state_47 + add_4
        hidden_state_47 = add_4 = None
        hidden_state_54 = torch.nn.functional.relu(hidden_state_53, inplace=False)
        hidden_state_55 = torch.conv2d(
            hidden_state_54,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_54 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_56 = torch.nn.functional.relu(hidden_state_55, inplace=False)
        hidden_state_55 = None
        hidden_state_57 = torch.conv2d(
            hidden_state_56,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_56 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_58 = hidden_state_57 + hidden_state_53
        hidden_state_57 = hidden_state_53 = None
        hidden_state_59 = torch.nn.functional.interpolate(
            hidden_state_58, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_58 = None
        hidden_state_60 = torch.conv2d(
            hidden_state_59,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_59 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_ = (None)
        hidden_state_62 = torch.nn.functional.relu(hidden_state_61, inplace=False)
        hidden_state_63 = torch.conv2d(
            hidden_state_62,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_62 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_ = (None)
        hidden_state_64 = torch.nn.functional.relu(hidden_state_63, inplace=False)
        hidden_state_63 = None
        hidden_state_65 = torch.conv2d(
            hidden_state_64,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_64 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_ = (None)
        add_7 = hidden_state_65 + hidden_state_61
        hidden_state_65 = hidden_state_61 = None
        hidden_state_66 = hidden_state_60 + add_7
        hidden_state_60 = add_7 = None
        hidden_state_67 = torch.nn.functional.relu(hidden_state_66, inplace=False)
        hidden_state_68 = torch.conv2d(
            hidden_state_67,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_67 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_ = (None)
        hidden_state_69 = torch.nn.functional.relu(hidden_state_68, inplace=False)
        hidden_state_68 = None
        hidden_state_70 = torch.conv2d(
            hidden_state_69,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_69 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_ = (None)
        hidden_state_71 = hidden_state_70 + hidden_state_66
        hidden_state_70 = hidden_state_66 = None
        hidden_state_72 = torch.nn.functional.interpolate(
            hidden_state_71, scale_factor=2, mode="bilinear", align_corners=True
        )
        hidden_state_71 = None
        hidden_state_73 = torch.conv2d(
            hidden_state_72,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_,
            l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_72 = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_ = l_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_ = (None)
        input_9 = torch.conv2d(
            hidden_state_73,
            l_self_modules_head_modules_head_modules_0_parameters_weight_,
            l_self_modules_head_modules_head_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hidden_state_73 = (
            l_self_modules_head_modules_head_modules_0_parameters_weight_
        ) = l_self_modules_head_modules_head_modules_0_parameters_bias_ = None
        input_10 = torch.nn.functional.interpolate(
            input_9, None, 2.0, "bilinear", True, recompute_scale_factor=None
        )
        input_9 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_head_modules_head_modules_2_parameters_weight_,
            l_self_modules_head_modules_head_modules_2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = (
            l_self_modules_head_modules_head_modules_2_parameters_weight_
        ) = l_self_modules_head_modules_head_modules_2_parameters_bias_ = None
        input_12 = torch.nn.functional.relu(input_11, inplace=False)
        input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_head_modules_head_modules_4_parameters_weight_,
            l_self_modules_head_modules_head_modules_4_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_12 = (
            l_self_modules_head_modules_head_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_head_modules_4_parameters_bias_ = None
        input_14 = torch.nn.functional.relu(input_13, inplace=False)
        input_13 = None
        predicted_depth = input_14.squeeze(dim=1)
        input_14 = None
        return (predicted_depth,)
