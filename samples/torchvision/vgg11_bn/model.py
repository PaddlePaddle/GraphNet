import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_19_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_19_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_19_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_23_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_23_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_23_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_23_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_25_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_25_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_26_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_26_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_26_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_26_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_1_buffers_running_mean_ = (
            L_self_modules_features_modules_1_buffers_running_mean_
        )
        l_self_modules_features_modules_1_buffers_running_var_ = (
            L_self_modules_features_modules_1_buffers_running_var_
        )
        l_self_modules_features_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_4_parameters_weight_ = (
            L_self_modules_features_modules_4_parameters_weight_
        )
        l_self_modules_features_modules_4_parameters_bias_ = (
            L_self_modules_features_modules_4_parameters_bias_
        )
        l_self_modules_features_modules_5_buffers_running_mean_ = (
            L_self_modules_features_modules_5_buffers_running_mean_
        )
        l_self_modules_features_modules_5_buffers_running_var_ = (
            L_self_modules_features_modules_5_buffers_running_var_
        )
        l_self_modules_features_modules_5_parameters_weight_ = (
            L_self_modules_features_modules_5_parameters_weight_
        )
        l_self_modules_features_modules_5_parameters_bias_ = (
            L_self_modules_features_modules_5_parameters_bias_
        )
        l_self_modules_features_modules_8_parameters_weight_ = (
            L_self_modules_features_modules_8_parameters_weight_
        )
        l_self_modules_features_modules_8_parameters_bias_ = (
            L_self_modules_features_modules_8_parameters_bias_
        )
        l_self_modules_features_modules_9_buffers_running_mean_ = (
            L_self_modules_features_modules_9_buffers_running_mean_
        )
        l_self_modules_features_modules_9_buffers_running_var_ = (
            L_self_modules_features_modules_9_buffers_running_var_
        )
        l_self_modules_features_modules_9_parameters_weight_ = (
            L_self_modules_features_modules_9_parameters_weight_
        )
        l_self_modules_features_modules_9_parameters_bias_ = (
            L_self_modules_features_modules_9_parameters_bias_
        )
        l_self_modules_features_modules_11_parameters_weight_ = (
            L_self_modules_features_modules_11_parameters_weight_
        )
        l_self_modules_features_modules_11_parameters_bias_ = (
            L_self_modules_features_modules_11_parameters_bias_
        )
        l_self_modules_features_modules_12_buffers_running_mean_ = (
            L_self_modules_features_modules_12_buffers_running_mean_
        )
        l_self_modules_features_modules_12_buffers_running_var_ = (
            L_self_modules_features_modules_12_buffers_running_var_
        )
        l_self_modules_features_modules_12_parameters_weight_ = (
            L_self_modules_features_modules_12_parameters_weight_
        )
        l_self_modules_features_modules_12_parameters_bias_ = (
            L_self_modules_features_modules_12_parameters_bias_
        )
        l_self_modules_features_modules_15_parameters_weight_ = (
            L_self_modules_features_modules_15_parameters_weight_
        )
        l_self_modules_features_modules_15_parameters_bias_ = (
            L_self_modules_features_modules_15_parameters_bias_
        )
        l_self_modules_features_modules_16_buffers_running_mean_ = (
            L_self_modules_features_modules_16_buffers_running_mean_
        )
        l_self_modules_features_modules_16_buffers_running_var_ = (
            L_self_modules_features_modules_16_buffers_running_var_
        )
        l_self_modules_features_modules_16_parameters_weight_ = (
            L_self_modules_features_modules_16_parameters_weight_
        )
        l_self_modules_features_modules_16_parameters_bias_ = (
            L_self_modules_features_modules_16_parameters_bias_
        )
        l_self_modules_features_modules_18_parameters_weight_ = (
            L_self_modules_features_modules_18_parameters_weight_
        )
        l_self_modules_features_modules_18_parameters_bias_ = (
            L_self_modules_features_modules_18_parameters_bias_
        )
        l_self_modules_features_modules_19_buffers_running_mean_ = (
            L_self_modules_features_modules_19_buffers_running_mean_
        )
        l_self_modules_features_modules_19_buffers_running_var_ = (
            L_self_modules_features_modules_19_buffers_running_var_
        )
        l_self_modules_features_modules_19_parameters_weight_ = (
            L_self_modules_features_modules_19_parameters_weight_
        )
        l_self_modules_features_modules_19_parameters_bias_ = (
            L_self_modules_features_modules_19_parameters_bias_
        )
        l_self_modules_features_modules_22_parameters_weight_ = (
            L_self_modules_features_modules_22_parameters_weight_
        )
        l_self_modules_features_modules_22_parameters_bias_ = (
            L_self_modules_features_modules_22_parameters_bias_
        )
        l_self_modules_features_modules_23_buffers_running_mean_ = (
            L_self_modules_features_modules_23_buffers_running_mean_
        )
        l_self_modules_features_modules_23_buffers_running_var_ = (
            L_self_modules_features_modules_23_buffers_running_var_
        )
        l_self_modules_features_modules_23_parameters_weight_ = (
            L_self_modules_features_modules_23_parameters_weight_
        )
        l_self_modules_features_modules_23_parameters_bias_ = (
            L_self_modules_features_modules_23_parameters_bias_
        )
        l_self_modules_features_modules_25_parameters_weight_ = (
            L_self_modules_features_modules_25_parameters_weight_
        )
        l_self_modules_features_modules_25_parameters_bias_ = (
            L_self_modules_features_modules_25_parameters_bias_
        )
        l_self_modules_features_modules_26_buffers_running_mean_ = (
            L_self_modules_features_modules_26_buffers_running_mean_
        )
        l_self_modules_features_modules_26_buffers_running_var_ = (
            L_self_modules_features_modules_26_buffers_running_var_
        )
        l_self_modules_features_modules_26_parameters_weight_ = (
            L_self_modules_features_modules_26_parameters_weight_
        )
        l_self_modules_features_modules_26_parameters_bias_ = (
            L_self_modules_features_modules_26_parameters_bias_
        )
        l_self_modules_classifier_modules_0_parameters_weight_ = (
            L_self_modules_classifier_modules_0_parameters_weight_
        )
        l_self_modules_classifier_modules_0_parameters_bias_ = (
            L_self_modules_classifier_modules_0_parameters_bias_
        )
        l_self_modules_classifier_modules_3_parameters_weight_ = (
            L_self_modules_classifier_modules_3_parameters_weight_
        )
        l_self_modules_classifier_modules_3_parameters_bias_ = (
            L_self_modules_classifier_modules_3_parameters_bias_
        )
        l_self_modules_classifier_modules_6_parameters_weight_ = (
            L_self_modules_classifier_modules_6_parameters_weight_
        )
        l_self_modules_classifier_modules_6_parameters_bias_ = (
            L_self_modules_classifier_modules_6_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_parameters_weight_,
            l_self_modules_features_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_features_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_features_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_features_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.nn.functional.max_pool2d(
            input_3, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_3 = None
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_features_modules_4_parameters_weight_,
            l_self_modules_features_modules_4_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = (
            l_self_modules_features_modules_4_parameters_weight_
        ) = l_self_modules_features_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_features_modules_5_buffers_running_mean_,
            l_self_modules_features_modules_5_buffers_running_var_,
            l_self_modules_features_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = (
            l_self_modules_features_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_5_buffers_running_var_
        ) = (
            l_self_modules_features_modules_5_parameters_weight_
        ) = l_self_modules_features_modules_5_parameters_bias_ = None
        input_7 = torch.nn.functional.relu(input_6, inplace=True)
        input_6 = None
        input_8 = torch.nn.functional.max_pool2d(
            input_7, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_7 = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_features_modules_8_parameters_weight_,
            l_self_modules_features_modules_8_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_8 = (
            l_self_modules_features_modules_8_parameters_weight_
        ) = l_self_modules_features_modules_8_parameters_bias_ = None
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_features_modules_9_buffers_running_mean_,
            l_self_modules_features_modules_9_buffers_running_var_,
            l_self_modules_features_modules_9_parameters_weight_,
            l_self_modules_features_modules_9_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = (
            l_self_modules_features_modules_9_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_9_buffers_running_var_
        ) = (
            l_self_modules_features_modules_9_parameters_weight_
        ) = l_self_modules_features_modules_9_parameters_bias_ = None
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_features_modules_11_parameters_weight_,
            l_self_modules_features_modules_11_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_11 = (
            l_self_modules_features_modules_11_parameters_weight_
        ) = l_self_modules_features_modules_11_parameters_bias_ = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_features_modules_12_buffers_running_mean_,
            l_self_modules_features_modules_12_buffers_running_var_,
            l_self_modules_features_modules_12_parameters_weight_,
            l_self_modules_features_modules_12_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = (
            l_self_modules_features_modules_12_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_12_buffers_running_var_
        ) = (
            l_self_modules_features_modules_12_parameters_weight_
        ) = l_self_modules_features_modules_12_parameters_bias_ = None
        input_14 = torch.nn.functional.relu(input_13, inplace=True)
        input_13 = None
        input_15 = torch.nn.functional.max_pool2d(
            input_14, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_14 = None
        input_16 = torch.conv2d(
            input_15,
            l_self_modules_features_modules_15_parameters_weight_,
            l_self_modules_features_modules_15_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_15 = (
            l_self_modules_features_modules_15_parameters_weight_
        ) = l_self_modules_features_modules_15_parameters_bias_ = None
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_features_modules_16_buffers_running_mean_,
            l_self_modules_features_modules_16_buffers_running_var_,
            l_self_modules_features_modules_16_parameters_weight_,
            l_self_modules_features_modules_16_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = (
            l_self_modules_features_modules_16_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_16_buffers_running_var_
        ) = (
            l_self_modules_features_modules_16_parameters_weight_
        ) = l_self_modules_features_modules_16_parameters_bias_ = None
        input_18 = torch.nn.functional.relu(input_17, inplace=True)
        input_17 = None
        input_19 = torch.conv2d(
            input_18,
            l_self_modules_features_modules_18_parameters_weight_,
            l_self_modules_features_modules_18_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_18 = (
            l_self_modules_features_modules_18_parameters_weight_
        ) = l_self_modules_features_modules_18_parameters_bias_ = None
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_features_modules_19_buffers_running_mean_,
            l_self_modules_features_modules_19_buffers_running_var_,
            l_self_modules_features_modules_19_parameters_weight_,
            l_self_modules_features_modules_19_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = (
            l_self_modules_features_modules_19_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_19_buffers_running_var_
        ) = (
            l_self_modules_features_modules_19_parameters_weight_
        ) = l_self_modules_features_modules_19_parameters_bias_ = None
        input_21 = torch.nn.functional.relu(input_20, inplace=True)
        input_20 = None
        input_22 = torch.nn.functional.max_pool2d(
            input_21, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_features_modules_22_parameters_weight_,
            l_self_modules_features_modules_22_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_22 = (
            l_self_modules_features_modules_22_parameters_weight_
        ) = l_self_modules_features_modules_22_parameters_bias_ = None
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_features_modules_23_buffers_running_mean_,
            l_self_modules_features_modules_23_buffers_running_var_,
            l_self_modules_features_modules_23_parameters_weight_,
            l_self_modules_features_modules_23_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = (
            l_self_modules_features_modules_23_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_23_buffers_running_var_
        ) = (
            l_self_modules_features_modules_23_parameters_weight_
        ) = l_self_modules_features_modules_23_parameters_bias_ = None
        input_25 = torch.nn.functional.relu(input_24, inplace=True)
        input_24 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_features_modules_25_parameters_weight_,
            l_self_modules_features_modules_25_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_25 = (
            l_self_modules_features_modules_25_parameters_weight_
        ) = l_self_modules_features_modules_25_parameters_bias_ = None
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_features_modules_26_buffers_running_mean_,
            l_self_modules_features_modules_26_buffers_running_var_,
            l_self_modules_features_modules_26_parameters_weight_,
            l_self_modules_features_modules_26_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = (
            l_self_modules_features_modules_26_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_26_buffers_running_var_
        ) = (
            l_self_modules_features_modules_26_parameters_weight_
        ) = l_self_modules_features_modules_26_parameters_bias_ = None
        input_28 = torch.nn.functional.relu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.nn.functional.max_pool2d(
            input_28, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_28 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_29, (7, 7))
        input_29 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_30 = torch._C._nn.linear(
            x_1,
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
        )
        x_1 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        input_31 = torch.nn.functional.relu(input_30, inplace=True)
        input_30 = None
        input_32 = torch.nn.functional.dropout(input_31, 0.5, False, False)
        input_31 = None
        input_33 = torch._C._nn.linear(
            input_32,
            l_self_modules_classifier_modules_3_parameters_weight_,
            l_self_modules_classifier_modules_3_parameters_bias_,
        )
        input_32 = (
            l_self_modules_classifier_modules_3_parameters_weight_
        ) = l_self_modules_classifier_modules_3_parameters_bias_ = None
        input_34 = torch.nn.functional.relu(input_33, inplace=True)
        input_33 = None
        input_35 = torch.nn.functional.dropout(input_34, 0.5, False, False)
        input_34 = None
        input_36 = torch._C._nn.linear(
            input_35,
            l_self_modules_classifier_modules_6_parameters_weight_,
            l_self_modules_classifier_modules_6_parameters_bias_,
        )
        input_35 = (
            l_self_modules_classifier_modules_6_parameters_weight_
        ) = l_self_modules_classifier_modules_6_parameters_bias_ = None
        return (input_36,)
