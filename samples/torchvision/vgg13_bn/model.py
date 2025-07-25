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
        L_self_modules_features_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_22_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_22_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_22_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_25_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_25_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_25_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_25_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_28_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_28_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_29_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_29_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_29_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_29_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_31_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_31_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_32_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_32_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_32_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_32_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_4_buffers_running_mean_ = (
            L_self_modules_features_modules_4_buffers_running_mean_
        )
        l_self_modules_features_modules_4_buffers_running_var_ = (
            L_self_modules_features_modules_4_buffers_running_var_
        )
        l_self_modules_features_modules_4_parameters_weight_ = (
            L_self_modules_features_modules_4_parameters_weight_
        )
        l_self_modules_features_modules_4_parameters_bias_ = (
            L_self_modules_features_modules_4_parameters_bias_
        )
        l_self_modules_features_modules_7_parameters_weight_ = (
            L_self_modules_features_modules_7_parameters_weight_
        )
        l_self_modules_features_modules_7_parameters_bias_ = (
            L_self_modules_features_modules_7_parameters_bias_
        )
        l_self_modules_features_modules_8_buffers_running_mean_ = (
            L_self_modules_features_modules_8_buffers_running_mean_
        )
        l_self_modules_features_modules_8_buffers_running_var_ = (
            L_self_modules_features_modules_8_buffers_running_var_
        )
        l_self_modules_features_modules_8_parameters_weight_ = (
            L_self_modules_features_modules_8_parameters_weight_
        )
        l_self_modules_features_modules_8_parameters_bias_ = (
            L_self_modules_features_modules_8_parameters_bias_
        )
        l_self_modules_features_modules_10_parameters_weight_ = (
            L_self_modules_features_modules_10_parameters_weight_
        )
        l_self_modules_features_modules_10_parameters_bias_ = (
            L_self_modules_features_modules_10_parameters_bias_
        )
        l_self_modules_features_modules_11_buffers_running_mean_ = (
            L_self_modules_features_modules_11_buffers_running_mean_
        )
        l_self_modules_features_modules_11_buffers_running_var_ = (
            L_self_modules_features_modules_11_buffers_running_var_
        )
        l_self_modules_features_modules_11_parameters_weight_ = (
            L_self_modules_features_modules_11_parameters_weight_
        )
        l_self_modules_features_modules_11_parameters_bias_ = (
            L_self_modules_features_modules_11_parameters_bias_
        )
        l_self_modules_features_modules_14_parameters_weight_ = (
            L_self_modules_features_modules_14_parameters_weight_
        )
        l_self_modules_features_modules_14_parameters_bias_ = (
            L_self_modules_features_modules_14_parameters_bias_
        )
        l_self_modules_features_modules_15_buffers_running_mean_ = (
            L_self_modules_features_modules_15_buffers_running_mean_
        )
        l_self_modules_features_modules_15_buffers_running_var_ = (
            L_self_modules_features_modules_15_buffers_running_var_
        )
        l_self_modules_features_modules_15_parameters_weight_ = (
            L_self_modules_features_modules_15_parameters_weight_
        )
        l_self_modules_features_modules_15_parameters_bias_ = (
            L_self_modules_features_modules_15_parameters_bias_
        )
        l_self_modules_features_modules_17_parameters_weight_ = (
            L_self_modules_features_modules_17_parameters_weight_
        )
        l_self_modules_features_modules_17_parameters_bias_ = (
            L_self_modules_features_modules_17_parameters_bias_
        )
        l_self_modules_features_modules_18_buffers_running_mean_ = (
            L_self_modules_features_modules_18_buffers_running_mean_
        )
        l_self_modules_features_modules_18_buffers_running_var_ = (
            L_self_modules_features_modules_18_buffers_running_var_
        )
        l_self_modules_features_modules_18_parameters_weight_ = (
            L_self_modules_features_modules_18_parameters_weight_
        )
        l_self_modules_features_modules_18_parameters_bias_ = (
            L_self_modules_features_modules_18_parameters_bias_
        )
        l_self_modules_features_modules_21_parameters_weight_ = (
            L_self_modules_features_modules_21_parameters_weight_
        )
        l_self_modules_features_modules_21_parameters_bias_ = (
            L_self_modules_features_modules_21_parameters_bias_
        )
        l_self_modules_features_modules_22_buffers_running_mean_ = (
            L_self_modules_features_modules_22_buffers_running_mean_
        )
        l_self_modules_features_modules_22_buffers_running_var_ = (
            L_self_modules_features_modules_22_buffers_running_var_
        )
        l_self_modules_features_modules_22_parameters_weight_ = (
            L_self_modules_features_modules_22_parameters_weight_
        )
        l_self_modules_features_modules_22_parameters_bias_ = (
            L_self_modules_features_modules_22_parameters_bias_
        )
        l_self_modules_features_modules_24_parameters_weight_ = (
            L_self_modules_features_modules_24_parameters_weight_
        )
        l_self_modules_features_modules_24_parameters_bias_ = (
            L_self_modules_features_modules_24_parameters_bias_
        )
        l_self_modules_features_modules_25_buffers_running_mean_ = (
            L_self_modules_features_modules_25_buffers_running_mean_
        )
        l_self_modules_features_modules_25_buffers_running_var_ = (
            L_self_modules_features_modules_25_buffers_running_var_
        )
        l_self_modules_features_modules_25_parameters_weight_ = (
            L_self_modules_features_modules_25_parameters_weight_
        )
        l_self_modules_features_modules_25_parameters_bias_ = (
            L_self_modules_features_modules_25_parameters_bias_
        )
        l_self_modules_features_modules_28_parameters_weight_ = (
            L_self_modules_features_modules_28_parameters_weight_
        )
        l_self_modules_features_modules_28_parameters_bias_ = (
            L_self_modules_features_modules_28_parameters_bias_
        )
        l_self_modules_features_modules_29_buffers_running_mean_ = (
            L_self_modules_features_modules_29_buffers_running_mean_
        )
        l_self_modules_features_modules_29_buffers_running_var_ = (
            L_self_modules_features_modules_29_buffers_running_var_
        )
        l_self_modules_features_modules_29_parameters_weight_ = (
            L_self_modules_features_modules_29_parameters_weight_
        )
        l_self_modules_features_modules_29_parameters_bias_ = (
            L_self_modules_features_modules_29_parameters_bias_
        )
        l_self_modules_features_modules_31_parameters_weight_ = (
            L_self_modules_features_modules_31_parameters_weight_
        )
        l_self_modules_features_modules_31_parameters_bias_ = (
            L_self_modules_features_modules_31_parameters_bias_
        )
        l_self_modules_features_modules_32_buffers_running_mean_ = (
            L_self_modules_features_modules_32_buffers_running_mean_
        )
        l_self_modules_features_modules_32_buffers_running_var_ = (
            L_self_modules_features_modules_32_buffers_running_var_
        )
        l_self_modules_features_modules_32_parameters_weight_ = (
            L_self_modules_features_modules_32_parameters_weight_
        )
        l_self_modules_features_modules_32_parameters_bias_ = (
            L_self_modules_features_modules_32_parameters_bias_
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
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_features_modules_3_parameters_weight_
        ) = l_self_modules_features_modules_3_parameters_bias_ = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_features_modules_4_buffers_running_mean_,
            l_self_modules_features_modules_4_buffers_running_var_,
            l_self_modules_features_modules_4_parameters_weight_,
            l_self_modules_features_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_features_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_4_buffers_running_var_
        ) = (
            l_self_modules_features_modules_4_parameters_weight_
        ) = l_self_modules_features_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.nn.functional.max_pool2d(
            input_6, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_6 = None
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_features_modules_7_parameters_weight_,
            l_self_modules_features_modules_7_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_7 = (
            l_self_modules_features_modules_7_parameters_weight_
        ) = l_self_modules_features_modules_7_parameters_bias_ = None
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_features_modules_8_buffers_running_mean_,
            l_self_modules_features_modules_8_buffers_running_var_,
            l_self_modules_features_modules_8_parameters_weight_,
            l_self_modules_features_modules_8_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = (
            l_self_modules_features_modules_8_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_8_buffers_running_var_
        ) = (
            l_self_modules_features_modules_8_parameters_weight_
        ) = l_self_modules_features_modules_8_parameters_bias_ = None
        input_10 = torch.nn.functional.relu(input_9, inplace=True)
        input_9 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_features_modules_10_parameters_weight_,
            l_self_modules_features_modules_10_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = (
            l_self_modules_features_modules_10_parameters_weight_
        ) = l_self_modules_features_modules_10_parameters_bias_ = None
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_features_modules_11_buffers_running_mean_,
            l_self_modules_features_modules_11_buffers_running_var_,
            l_self_modules_features_modules_11_parameters_weight_,
            l_self_modules_features_modules_11_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = (
            l_self_modules_features_modules_11_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_11_buffers_running_var_
        ) = (
            l_self_modules_features_modules_11_parameters_weight_
        ) = l_self_modules_features_modules_11_parameters_bias_ = None
        input_13 = torch.nn.functional.relu(input_12, inplace=True)
        input_12 = None
        input_14 = torch.nn.functional.max_pool2d(
            input_13, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_13 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_features_modules_14_parameters_weight_,
            l_self_modules_features_modules_14_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_14 = (
            l_self_modules_features_modules_14_parameters_weight_
        ) = l_self_modules_features_modules_14_parameters_bias_ = None
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_features_modules_15_buffers_running_mean_,
            l_self_modules_features_modules_15_buffers_running_var_,
            l_self_modules_features_modules_15_parameters_weight_,
            l_self_modules_features_modules_15_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = (
            l_self_modules_features_modules_15_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_15_buffers_running_var_
        ) = (
            l_self_modules_features_modules_15_parameters_weight_
        ) = l_self_modules_features_modules_15_parameters_bias_ = None
        input_17 = torch.nn.functional.relu(input_16, inplace=True)
        input_16 = None
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_features_modules_17_parameters_weight_,
            l_self_modules_features_modules_17_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_17 = (
            l_self_modules_features_modules_17_parameters_weight_
        ) = l_self_modules_features_modules_17_parameters_bias_ = None
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_features_modules_18_buffers_running_mean_,
            l_self_modules_features_modules_18_buffers_running_var_,
            l_self_modules_features_modules_18_parameters_weight_,
            l_self_modules_features_modules_18_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = (
            l_self_modules_features_modules_18_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_18_buffers_running_var_
        ) = (
            l_self_modules_features_modules_18_parameters_weight_
        ) = l_self_modules_features_modules_18_parameters_bias_ = None
        input_20 = torch.nn.functional.relu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.nn.functional.max_pool2d(
            input_20, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_20 = None
        input_22 = torch.conv2d(
            input_21,
            l_self_modules_features_modules_21_parameters_weight_,
            l_self_modules_features_modules_21_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_21 = (
            l_self_modules_features_modules_21_parameters_weight_
        ) = l_self_modules_features_modules_21_parameters_bias_ = None
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_features_modules_22_buffers_running_mean_,
            l_self_modules_features_modules_22_buffers_running_var_,
            l_self_modules_features_modules_22_parameters_weight_,
            l_self_modules_features_modules_22_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_22 = (
            l_self_modules_features_modules_22_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_22_buffers_running_var_
        ) = (
            l_self_modules_features_modules_22_parameters_weight_
        ) = l_self_modules_features_modules_22_parameters_bias_ = None
        input_24 = torch.nn.functional.relu(input_23, inplace=True)
        input_23 = None
        input_25 = torch.conv2d(
            input_24,
            l_self_modules_features_modules_24_parameters_weight_,
            l_self_modules_features_modules_24_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_24 = (
            l_self_modules_features_modules_24_parameters_weight_
        ) = l_self_modules_features_modules_24_parameters_bias_ = None
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_features_modules_25_buffers_running_mean_,
            l_self_modules_features_modules_25_buffers_running_var_,
            l_self_modules_features_modules_25_parameters_weight_,
            l_self_modules_features_modules_25_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = (
            l_self_modules_features_modules_25_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_25_buffers_running_var_
        ) = (
            l_self_modules_features_modules_25_parameters_weight_
        ) = l_self_modules_features_modules_25_parameters_bias_ = None
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.nn.functional.max_pool2d(
            input_27, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_features_modules_28_parameters_weight_,
            l_self_modules_features_modules_28_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_28 = (
            l_self_modules_features_modules_28_parameters_weight_
        ) = l_self_modules_features_modules_28_parameters_bias_ = None
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_features_modules_29_buffers_running_mean_,
            l_self_modules_features_modules_29_buffers_running_var_,
            l_self_modules_features_modules_29_parameters_weight_,
            l_self_modules_features_modules_29_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = (
            l_self_modules_features_modules_29_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_29_buffers_running_var_
        ) = (
            l_self_modules_features_modules_29_parameters_weight_
        ) = l_self_modules_features_modules_29_parameters_bias_ = None
        input_31 = torch.nn.functional.relu(input_30, inplace=True)
        input_30 = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_features_modules_31_parameters_weight_,
            l_self_modules_features_modules_31_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_31 = (
            l_self_modules_features_modules_31_parameters_weight_
        ) = l_self_modules_features_modules_31_parameters_bias_ = None
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_features_modules_32_buffers_running_mean_,
            l_self_modules_features_modules_32_buffers_running_var_,
            l_self_modules_features_modules_32_parameters_weight_,
            l_self_modules_features_modules_32_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_32 = (
            l_self_modules_features_modules_32_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_32_buffers_running_var_
        ) = (
            l_self_modules_features_modules_32_parameters_weight_
        ) = l_self_modules_features_modules_32_parameters_bias_ = None
        input_34 = torch.nn.functional.relu(input_33, inplace=True)
        input_33 = None
        input_35 = torch.nn.functional.max_pool2d(
            input_34, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_34 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_35, (7, 7))
        input_35 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_36 = torch._C._nn.linear(
            x_1,
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
        )
        x_1 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.nn.functional.dropout(input_37, 0.5, False, False)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_classifier_modules_3_parameters_weight_,
            l_self_modules_classifier_modules_3_parameters_bias_,
        )
        input_38 = (
            l_self_modules_classifier_modules_3_parameters_weight_
        ) = l_self_modules_classifier_modules_3_parameters_bias_ = None
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.nn.functional.dropout(input_40, 0.5, False, False)
        input_40 = None
        input_42 = torch._C._nn.linear(
            input_41,
            l_self_modules_classifier_modules_6_parameters_weight_,
            l_self_modules_classifier_modules_6_parameters_bias_,
        )
        input_41 = (
            l_self_modules_classifier_modules_6_parameters_weight_
        ) = l_self_modules_classifier_modules_6_parameters_bias_ = None
        return (input_42,)
