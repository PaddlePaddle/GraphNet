import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_10_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_10_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_features_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_features_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_features_modules_0_parameters_bias_ = (
            L_self_modules_backbone_modules_features_modules_0_parameters_bias_
        )
        l_self_modules_backbone_modules_features_modules_3_parameters_weight_ = (
            L_self_modules_backbone_modules_features_modules_3_parameters_weight_
        )
        l_self_modules_backbone_modules_features_modules_3_parameters_bias_ = (
            L_self_modules_backbone_modules_features_modules_3_parameters_bias_
        )
        l_self_modules_backbone_modules_features_modules_6_parameters_weight_ = (
            L_self_modules_backbone_modules_features_modules_6_parameters_weight_
        )
        l_self_modules_backbone_modules_features_modules_6_parameters_bias_ = (
            L_self_modules_backbone_modules_features_modules_6_parameters_bias_
        )
        l_self_modules_backbone_modules_features_modules_8_parameters_weight_ = (
            L_self_modules_backbone_modules_features_modules_8_parameters_weight_
        )
        l_self_modules_backbone_modules_features_modules_8_parameters_bias_ = (
            L_self_modules_backbone_modules_features_modules_8_parameters_bias_
        )
        l_self_modules_backbone_modules_features_modules_10_parameters_weight_ = (
            L_self_modules_backbone_modules_features_modules_10_parameters_weight_
        )
        l_self_modules_backbone_modules_features_modules_10_parameters_bias_ = (
            L_self_modules_backbone_modules_features_modules_10_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_
        )
        l_self_modules_head_modules_final_layer_parameters_weight_ = (
            L_self_modules_head_modules_final_layer_parameters_weight_
        )
        l_self_modules_head_modules_final_layer_parameters_bias_ = (
            L_self_modules_head_modules_final_layer_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_features_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_0_parameters_bias_,
            (4, 4),
            (2, 2),
            (1, 1),
            1,
        )
        l_inputs_ = (
            l_self_modules_backbone_modules_features_modules_0_parameters_weight_
        ) = l_self_modules_backbone_modules_features_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.nn.functional.max_pool2d(
            input_2, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_features_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_3_parameters_bias_,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_backbone_modules_features_modules_3_parameters_weight_
        ) = l_self_modules_backbone_modules_features_modules_3_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch.nn.functional.max_pool2d(
            input_5, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_features_modules_6_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_6_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = (
            l_self_modules_backbone_modules_features_modules_6_parameters_weight_
        ) = l_self_modules_backbone_modules_features_modules_6_parameters_bias_ = None
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
        input_7 = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_backbone_modules_features_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_8_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_8 = (
            l_self_modules_backbone_modules_features_modules_8_parameters_weight_
        ) = l_self_modules_backbone_modules_features_modules_8_parameters_bias_ = None
        input_10 = torch.nn.functional.relu(input_9, inplace=True)
        input_9 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_backbone_modules_features_modules_10_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_10_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = (
            l_self_modules_backbone_modules_features_modules_10_parameters_weight_
        ) = l_self_modules_backbone_modules_features_modules_10_parameters_bias_ = None
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        input_13 = torch.nn.functional.max_pool2d(
            input_12, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_12 = None
        input_14 = torch.conv_transpose2d(
            input_13,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_13 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        input_17 = torch.conv_transpose2d(
            input_16,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_16 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        input_20 = torch.conv_transpose2d(
            input_19,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_19 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_22 = torch.nn.functional.relu(input_21, inplace=True)
        input_21 = None
        x = torch.conv2d(
            input_22,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x,)
