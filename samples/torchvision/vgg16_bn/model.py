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
        L_self_modules_features_modules_20_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_21_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_21_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_21_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_24_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_24_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_25_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_25_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_25_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_25_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_27_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_27_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_28_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_28_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_28_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_28_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_30_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_30_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_31_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_31_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_31_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_31_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_34_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_34_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_35_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_35_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_35_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_35_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_37_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_37_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_38_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_38_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_38_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_38_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_40_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_40_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_41_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_41_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_41_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_41_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_20_parameters_weight_ = (
            L_self_modules_features_modules_20_parameters_weight_
        )
        l_self_modules_features_modules_20_parameters_bias_ = (
            L_self_modules_features_modules_20_parameters_bias_
        )
        l_self_modules_features_modules_21_buffers_running_mean_ = (
            L_self_modules_features_modules_21_buffers_running_mean_
        )
        l_self_modules_features_modules_21_buffers_running_var_ = (
            L_self_modules_features_modules_21_buffers_running_var_
        )
        l_self_modules_features_modules_21_parameters_weight_ = (
            L_self_modules_features_modules_21_parameters_weight_
        )
        l_self_modules_features_modules_21_parameters_bias_ = (
            L_self_modules_features_modules_21_parameters_bias_
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
        l_self_modules_features_modules_27_parameters_weight_ = (
            L_self_modules_features_modules_27_parameters_weight_
        )
        l_self_modules_features_modules_27_parameters_bias_ = (
            L_self_modules_features_modules_27_parameters_bias_
        )
        l_self_modules_features_modules_28_buffers_running_mean_ = (
            L_self_modules_features_modules_28_buffers_running_mean_
        )
        l_self_modules_features_modules_28_buffers_running_var_ = (
            L_self_modules_features_modules_28_buffers_running_var_
        )
        l_self_modules_features_modules_28_parameters_weight_ = (
            L_self_modules_features_modules_28_parameters_weight_
        )
        l_self_modules_features_modules_28_parameters_bias_ = (
            L_self_modules_features_modules_28_parameters_bias_
        )
        l_self_modules_features_modules_30_parameters_weight_ = (
            L_self_modules_features_modules_30_parameters_weight_
        )
        l_self_modules_features_modules_30_parameters_bias_ = (
            L_self_modules_features_modules_30_parameters_bias_
        )
        l_self_modules_features_modules_31_buffers_running_mean_ = (
            L_self_modules_features_modules_31_buffers_running_mean_
        )
        l_self_modules_features_modules_31_buffers_running_var_ = (
            L_self_modules_features_modules_31_buffers_running_var_
        )
        l_self_modules_features_modules_31_parameters_weight_ = (
            L_self_modules_features_modules_31_parameters_weight_
        )
        l_self_modules_features_modules_31_parameters_bias_ = (
            L_self_modules_features_modules_31_parameters_bias_
        )
        l_self_modules_features_modules_34_parameters_weight_ = (
            L_self_modules_features_modules_34_parameters_weight_
        )
        l_self_modules_features_modules_34_parameters_bias_ = (
            L_self_modules_features_modules_34_parameters_bias_
        )
        l_self_modules_features_modules_35_buffers_running_mean_ = (
            L_self_modules_features_modules_35_buffers_running_mean_
        )
        l_self_modules_features_modules_35_buffers_running_var_ = (
            L_self_modules_features_modules_35_buffers_running_var_
        )
        l_self_modules_features_modules_35_parameters_weight_ = (
            L_self_modules_features_modules_35_parameters_weight_
        )
        l_self_modules_features_modules_35_parameters_bias_ = (
            L_self_modules_features_modules_35_parameters_bias_
        )
        l_self_modules_features_modules_37_parameters_weight_ = (
            L_self_modules_features_modules_37_parameters_weight_
        )
        l_self_modules_features_modules_37_parameters_bias_ = (
            L_self_modules_features_modules_37_parameters_bias_
        )
        l_self_modules_features_modules_38_buffers_running_mean_ = (
            L_self_modules_features_modules_38_buffers_running_mean_
        )
        l_self_modules_features_modules_38_buffers_running_var_ = (
            L_self_modules_features_modules_38_buffers_running_var_
        )
        l_self_modules_features_modules_38_parameters_weight_ = (
            L_self_modules_features_modules_38_parameters_weight_
        )
        l_self_modules_features_modules_38_parameters_bias_ = (
            L_self_modules_features_modules_38_parameters_bias_
        )
        l_self_modules_features_modules_40_parameters_weight_ = (
            L_self_modules_features_modules_40_parameters_weight_
        )
        l_self_modules_features_modules_40_parameters_bias_ = (
            L_self_modules_features_modules_40_parameters_bias_
        )
        l_self_modules_features_modules_41_buffers_running_mean_ = (
            L_self_modules_features_modules_41_buffers_running_mean_
        )
        l_self_modules_features_modules_41_buffers_running_var_ = (
            L_self_modules_features_modules_41_buffers_running_var_
        )
        l_self_modules_features_modules_41_parameters_weight_ = (
            L_self_modules_features_modules_41_parameters_weight_
        )
        l_self_modules_features_modules_41_parameters_bias_ = (
            L_self_modules_features_modules_41_parameters_bias_
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
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_features_modules_20_parameters_weight_,
            l_self_modules_features_modules_20_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_20 = (
            l_self_modules_features_modules_20_parameters_weight_
        ) = l_self_modules_features_modules_20_parameters_bias_ = None
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_features_modules_21_buffers_running_mean_,
            l_self_modules_features_modules_21_buffers_running_var_,
            l_self_modules_features_modules_21_parameters_weight_,
            l_self_modules_features_modules_21_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_21 = (
            l_self_modules_features_modules_21_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_21_buffers_running_var_
        ) = (
            l_self_modules_features_modules_21_parameters_weight_
        ) = l_self_modules_features_modules_21_parameters_bias_ = None
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.nn.functional.max_pool2d(
            input_23, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
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
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_features_modules_27_parameters_weight_,
            l_self_modules_features_modules_27_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_27 = (
            l_self_modules_features_modules_27_parameters_weight_
        ) = l_self_modules_features_modules_27_parameters_bias_ = None
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_features_modules_28_buffers_running_mean_,
            l_self_modules_features_modules_28_buffers_running_var_,
            l_self_modules_features_modules_28_parameters_weight_,
            l_self_modules_features_modules_28_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = (
            l_self_modules_features_modules_28_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_28_buffers_running_var_
        ) = (
            l_self_modules_features_modules_28_parameters_weight_
        ) = l_self_modules_features_modules_28_parameters_bias_ = None
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_features_modules_30_parameters_weight_,
            l_self_modules_features_modules_30_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_30 = (
            l_self_modules_features_modules_30_parameters_weight_
        ) = l_self_modules_features_modules_30_parameters_bias_ = None
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_features_modules_31_buffers_running_mean_,
            l_self_modules_features_modules_31_buffers_running_var_,
            l_self_modules_features_modules_31_parameters_weight_,
            l_self_modules_features_modules_31_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = (
            l_self_modules_features_modules_31_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_31_buffers_running_var_
        ) = (
            l_self_modules_features_modules_31_parameters_weight_
        ) = l_self_modules_features_modules_31_parameters_bias_ = None
        input_33 = torch.nn.functional.relu(input_32, inplace=True)
        input_32 = None
        input_34 = torch.nn.functional.max_pool2d(
            input_33, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_33 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_features_modules_34_parameters_weight_,
            l_self_modules_features_modules_34_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_34 = (
            l_self_modules_features_modules_34_parameters_weight_
        ) = l_self_modules_features_modules_34_parameters_bias_ = None
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_features_modules_35_buffers_running_mean_,
            l_self_modules_features_modules_35_buffers_running_var_,
            l_self_modules_features_modules_35_parameters_weight_,
            l_self_modules_features_modules_35_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = (
            l_self_modules_features_modules_35_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_35_buffers_running_var_
        ) = (
            l_self_modules_features_modules_35_parameters_weight_
        ) = l_self_modules_features_modules_35_parameters_bias_ = None
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_features_modules_37_parameters_weight_,
            l_self_modules_features_modules_37_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_37 = (
            l_self_modules_features_modules_37_parameters_weight_
        ) = l_self_modules_features_modules_37_parameters_bias_ = None
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_features_modules_38_buffers_running_mean_,
            l_self_modules_features_modules_38_buffers_running_var_,
            l_self_modules_features_modules_38_parameters_weight_,
            l_self_modules_features_modules_38_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = (
            l_self_modules_features_modules_38_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_38_buffers_running_var_
        ) = (
            l_self_modules_features_modules_38_parameters_weight_
        ) = l_self_modules_features_modules_38_parameters_bias_ = None
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_features_modules_40_parameters_weight_,
            l_self_modules_features_modules_40_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_40 = (
            l_self_modules_features_modules_40_parameters_weight_
        ) = l_self_modules_features_modules_40_parameters_bias_ = None
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_features_modules_41_buffers_running_mean_,
            l_self_modules_features_modules_41_buffers_running_var_,
            l_self_modules_features_modules_41_parameters_weight_,
            l_self_modules_features_modules_41_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = (
            l_self_modules_features_modules_41_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_41_buffers_running_var_
        ) = (
            l_self_modules_features_modules_41_parameters_weight_
        ) = l_self_modules_features_modules_41_parameters_bias_ = None
        input_43 = torch.nn.functional.relu(input_42, inplace=True)
        input_42 = None
        input_44 = torch.nn.functional.max_pool2d(
            input_43, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_43 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_44, (7, 7))
        input_44 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_45 = torch._C._nn.linear(
            x_1,
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
        )
        x_1 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        input_46 = torch.nn.functional.relu(input_45, inplace=True)
        input_45 = None
        input_47 = torch.nn.functional.dropout(input_46, 0.5, False, False)
        input_46 = None
        input_48 = torch._C._nn.linear(
            input_47,
            l_self_modules_classifier_modules_3_parameters_weight_,
            l_self_modules_classifier_modules_3_parameters_bias_,
        )
        input_47 = (
            l_self_modules_classifier_modules_3_parameters_weight_
        ) = l_self_modules_classifier_modules_3_parameters_bias_ = None
        input_49 = torch.nn.functional.relu(input_48, inplace=True)
        input_48 = None
        input_50 = torch.nn.functional.dropout(input_49, 0.5, False, False)
        input_49 = None
        input_51 = torch._C._nn.linear(
            input_50,
            l_self_modules_classifier_modules_6_parameters_weight_,
            l_self_modules_classifier_modules_6_parameters_bias_,
        )
        input_50 = (
            l_self_modules_classifier_modules_6_parameters_weight_
        ) = l_self_modules_classifier_modules_6_parameters_bias_ = None
        return (input_51,)
