import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_0_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_1_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_2_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_stack0_feature_maps_3_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_auxiliary_head_modules_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_auxiliary_head_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_auxiliary_head_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_
        l_stack0_feature_maps_0_ = L_stack0_feature_maps_0_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_
        l_stack0_feature_maps_1_ = L_stack0_feature_maps_1_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_
        l_stack0_feature_maps_2_ = L_stack0_feature_maps_2_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_bias_
        l_stack0_feature_maps_3_ = L_stack0_feature_maps_3_
        l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_mean_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_mean_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_var_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_var_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_weight_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_weight_
        l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_bias_ = L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_bias_
        l_self_modules_decode_head_modules_classifier_parameters_weight_ = (
            L_self_modules_decode_head_modules_classifier_parameters_weight_
        )
        l_self_modules_decode_head_modules_classifier_parameters_bias_ = (
            L_self_modules_decode_head_modules_classifier_parameters_bias_
        )
        l_self_modules_auxiliary_head_modules_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_auxiliary_head_modules_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_mean_ = L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_mean_
        l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_var_ = L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_var_
        l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_weight_ = L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_weight_
        l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_bias_ = L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_bias_
        l_self_modules_auxiliary_head_modules_classifier_parameters_weight_ = (
            L_self_modules_auxiliary_head_modules_classifier_parameters_weight_
        )
        l_self_modules_auxiliary_head_modules_classifier_parameters_bias_ = (
            L_self_modules_auxiliary_head_modules_classifier_parameters_bias_
        )
        output = torch.conv2d(
            l_stack0_feature_maps_0_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_0_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_ = (None)
        output_1 = torch.nn.functional.batch_norm(
            output,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_bias_ = (None)
        output_2 = torch.nn.functional.relu(output_1, inplace=False)
        output_1 = None
        output_3 = torch.conv2d(
            l_stack0_feature_maps_1_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_1_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_ = (None)
        output_4 = torch.nn.functional.batch_norm(
            output_3,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_3 = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_bias_ = (None)
        output_5 = torch.nn.functional.relu(output_4, inplace=False)
        output_4 = None
        output_6 = torch.conv2d(
            l_stack0_feature_maps_2_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        output_7 = torch.nn.functional.batch_norm(
            output_6,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_6 = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_bias_ = (None)
        output_8 = torch.nn.functional.relu(output_7, inplace=False)
        output_7 = None
        hidden_state = torch.nn.functional.adaptive_avg_pool2d(
            l_stack0_feature_maps_3_, 1
        )
        output_9 = torch.conv2d(
            hidden_state,
            l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state = l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_conv_parameters_weight_ = (None)
        output_10 = torch.nn.functional.batch_norm(
            output_9,
            l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_9 = l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_bias_ = (None)
        output_11 = torch.nn.functional.relu(output_10, inplace=False)
        output_10 = None
        upsampled_ppm_out = torch.nn.functional.interpolate(
            output_11, size=(16, 16), mode="bilinear", align_corners=False
        )
        output_11 = None
        hidden_state_1 = torch.nn.functional.adaptive_avg_pool2d(
            l_stack0_feature_maps_3_, 2
        )
        output_12 = torch.conv2d(
            hidden_state_1,
            l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_1 = l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_conv_parameters_weight_ = (None)
        output_13 = torch.nn.functional.batch_norm(
            output_12,
            l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_12 = l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_bias_ = (None)
        output_14 = torch.nn.functional.relu(output_13, inplace=False)
        output_13 = None
        upsampled_ppm_out_1 = torch.nn.functional.interpolate(
            output_14, size=(16, 16), mode="bilinear", align_corners=False
        )
        output_14 = None
        hidden_state_2 = torch.nn.functional.adaptive_avg_pool2d(
            l_stack0_feature_maps_3_, 3
        )
        output_15 = torch.conv2d(
            hidden_state_2,
            l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_2 = l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_conv_parameters_weight_ = (None)
        output_16 = torch.nn.functional.batch_norm(
            output_15,
            l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_15 = l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_bias_ = (None)
        output_17 = torch.nn.functional.relu(output_16, inplace=False)
        output_16 = None
        upsampled_ppm_out_2 = torch.nn.functional.interpolate(
            output_17, size=(16, 16), mode="bilinear", align_corners=False
        )
        output_17 = None
        hidden_state_3 = torch.nn.functional.adaptive_avg_pool2d(
            l_stack0_feature_maps_3_, 6
        )
        output_18 = torch.conv2d(
            hidden_state_3,
            l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        hidden_state_3 = l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_conv_parameters_weight_ = (None)
        output_19 = torch.nn.functional.batch_norm(
            output_18,
            l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_18 = l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_bias_ = (None)
        output_20 = torch.nn.functional.relu(output_19, inplace=False)
        output_19 = None
        upsampled_ppm_out_3 = torch.nn.functional.interpolate(
            output_20, size=(16, 16), mode="bilinear", align_corners=False
        )
        output_20 = None
        psp_outs = torch.cat(
            [
                l_stack0_feature_maps_3_,
                upsampled_ppm_out,
                upsampled_ppm_out_1,
                upsampled_ppm_out_2,
                upsampled_ppm_out_3,
            ],
            dim=1,
        )
        l_stack0_feature_maps_3_ = (
            upsampled_ppm_out
        ) = upsampled_ppm_out_1 = upsampled_ppm_out_2 = upsampled_ppm_out_3 = None
        output_21 = torch.conv2d(
            psp_outs,
            l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        psp_outs = l_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_ = (None)
        output_22 = torch.nn.functional.batch_norm(
            output_21,
            l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_21 = l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_bias_ = (None)
        output_23 = torch.nn.functional.relu(output_22, inplace=False)
        output_22 = None
        interpolate_4 = torch.nn.functional.interpolate(
            output_23, size=(32, 32), mode="bilinear", align_corners=False
        )
        add = output_8 + interpolate_4
        output_8 = interpolate_4 = None
        interpolate_5 = torch.nn.functional.interpolate(
            add, size=(64, 64), mode="bilinear", align_corners=False
        )
        add_1 = output_5 + interpolate_5
        output_5 = interpolate_5 = None
        interpolate_6 = torch.nn.functional.interpolate(
            add_1, size=(128, 128), mode="bilinear", align_corners=False
        )
        add_2 = output_2 + interpolate_6
        output_2 = interpolate_6 = None
        output_24 = torch.conv2d(
            add_2,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_2 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_ = (None)
        output_25 = torch.nn.functional.batch_norm(
            output_24,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_24 = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_bias_ = (None)
        output_26 = torch.nn.functional.relu(output_25, inplace=False)
        output_25 = None
        output_27 = torch.conv2d(
            add_1,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add_1 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_ = (None)
        output_28 = torch.nn.functional.batch_norm(
            output_27,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_27 = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_bias_ = (None)
        output_29 = torch.nn.functional.relu(output_28, inplace=False)
        output_28 = None
        output_30 = torch.conv2d(
            add,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        add = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_ = (None)
        output_31 = torch.nn.functional.batch_norm(
            output_30,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_30 = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_bias_ = (None)
        output_32 = torch.nn.functional.relu(output_31, inplace=False)
        output_31 = None
        interpolate_7 = torch.nn.functional.interpolate(
            output_23, size=(128, 128), mode="bilinear", align_corners=False
        )
        output_23 = None
        interpolate_8 = torch.nn.functional.interpolate(
            output_32, size=(128, 128), mode="bilinear", align_corners=False
        )
        output_32 = None
        interpolate_9 = torch.nn.functional.interpolate(
            output_29, size=(128, 128), mode="bilinear", align_corners=False
        )
        output_29 = None
        fpn_outs = torch.cat(
            [output_26, interpolate_9, interpolate_8, interpolate_7], dim=1
        )
        output_26 = interpolate_9 = interpolate_8 = interpolate_7 = None
        output_33 = torch.conv2d(
            fpn_outs,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        fpn_outs = l_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_ = (None)
        output_34 = torch.nn.functional.batch_norm(
            output_33,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_weight_,
            l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_33 = l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_mean_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_var_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_weight_ = l_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_bias_ = (None)
        output_35 = torch.nn.functional.relu(output_34, inplace=False)
        output_34 = None
        output_36 = torch.conv2d(
            output_35,
            l_self_modules_decode_head_modules_classifier_parameters_weight_,
            l_self_modules_decode_head_modules_classifier_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        output_35 = (
            l_self_modules_decode_head_modules_classifier_parameters_weight_
        ) = l_self_modules_decode_head_modules_classifier_parameters_bias_ = None
        logits = torch.nn.functional.interpolate(
            output_36, size=(512, 512), mode="bilinear", align_corners=False
        )
        output_36 = None
        output_37 = torch.conv2d(
            l_stack0_feature_maps_2_,
            l_self_modules_auxiliary_head_modules_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_stack0_feature_maps_2_ = l_self_modules_auxiliary_head_modules_convs_modules_0_modules_conv_parameters_weight_ = (None)
        output_38 = torch.nn.functional.batch_norm(
            output_37,
            l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_mean_,
            l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_var_,
            l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_weight_,
            l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        output_37 = l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_mean_ = l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_var_ = l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_weight_ = l_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_bias_ = (None)
        output_39 = torch.nn.functional.relu(output_38, inplace=False)
        output_38 = None
        output_40 = torch.conv2d(
            output_39,
            l_self_modules_auxiliary_head_modules_classifier_parameters_weight_,
            l_self_modules_auxiliary_head_modules_classifier_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        output_39 = (
            l_self_modules_auxiliary_head_modules_classifier_parameters_weight_
        ) = l_self_modules_auxiliary_head_modules_classifier_parameters_bias_ = None
        auxiliary_logits = torch.nn.functional.interpolate(
            output_40, size=(512, 512), mode="bilinear", align_corners=False
        )
        output_40 = auxiliary_logits = None
        return (logits,)
