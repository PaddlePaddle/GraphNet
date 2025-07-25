import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_20_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_22_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_22_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_2_parameters_bias_ = (
            L_self_modules_features_modules_2_parameters_bias_
        )
        l_self_modules_features_modules_5_parameters_weight_ = (
            L_self_modules_features_modules_5_parameters_weight_
        )
        l_self_modules_features_modules_5_parameters_bias_ = (
            L_self_modules_features_modules_5_parameters_bias_
        )
        l_self_modules_features_modules_7_parameters_weight_ = (
            L_self_modules_features_modules_7_parameters_weight_
        )
        l_self_modules_features_modules_7_parameters_bias_ = (
            L_self_modules_features_modules_7_parameters_bias_
        )
        l_self_modules_features_modules_10_parameters_weight_ = (
            L_self_modules_features_modules_10_parameters_weight_
        )
        l_self_modules_features_modules_10_parameters_bias_ = (
            L_self_modules_features_modules_10_parameters_bias_
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
        l_self_modules_features_modules_17_parameters_weight_ = (
            L_self_modules_features_modules_17_parameters_weight_
        )
        l_self_modules_features_modules_17_parameters_bias_ = (
            L_self_modules_features_modules_17_parameters_bias_
        )
        l_self_modules_features_modules_20_parameters_weight_ = (
            L_self_modules_features_modules_20_parameters_weight_
        )
        l_self_modules_features_modules_20_parameters_bias_ = (
            L_self_modules_features_modules_20_parameters_bias_
        )
        l_self_modules_features_modules_22_parameters_weight_ = (
            L_self_modules_features_modules_22_parameters_weight_
        )
        l_self_modules_features_modules_22_parameters_bias_ = (
            L_self_modules_features_modules_22_parameters_bias_
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
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.conv2d(
            input_2,
            l_self_modules_features_modules_2_parameters_weight_,
            l_self_modules_features_modules_2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = (
            l_self_modules_features_modules_2_parameters_weight_
        ) = l_self_modules_features_modules_2_parameters_bias_ = None
        input_4 = torch.nn.functional.relu(input_3, inplace=True)
        input_3 = None
        input_5 = torch.nn.functional.max_pool2d(
            input_4, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_4 = None
        input_6 = torch.conv2d(
            input_5,
            l_self_modules_features_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_5 = (
            l_self_modules_features_modules_5_parameters_weight_
        ) = l_self_modules_features_modules_5_parameters_bias_ = None
        input_7 = torch.nn.functional.relu(input_6, inplace=True)
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
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        input_10 = torch.nn.functional.max_pool2d(
            input_9, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
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
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_features_modules_12_parameters_weight_,
            l_self_modules_features_modules_12_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_12 = (
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
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        input_20 = torch.nn.functional.max_pool2d(
            input_19, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
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
        input_22 = torch.nn.functional.relu(input_21, inplace=True)
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
        input_24 = torch.nn.functional.relu(input_23, inplace=True)
        input_23 = None
        input_25 = torch.nn.functional.max_pool2d(
            input_24, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_24 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_25, (7, 7))
        input_25 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_26 = torch._C._nn.linear(
            x_1,
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
        )
        x_1 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.nn.functional.dropout(input_27, 0.5, False, False)
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_classifier_modules_3_parameters_weight_,
            l_self_modules_classifier_modules_3_parameters_bias_,
        )
        input_28 = (
            l_self_modules_classifier_modules_3_parameters_weight_
        ) = l_self_modules_classifier_modules_3_parameters_bias_ = None
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.nn.functional.dropout(input_30, 0.5, False, False)
        input_30 = None
        input_32 = torch._C._nn.linear(
            input_31,
            l_self_modules_classifier_modules_6_parameters_weight_,
            l_self_modules_classifier_modules_6_parameters_bias_,
        )
        input_31 = (
            l_self_modules_classifier_modules_6_parameters_weight_
        ) = l_self_modules_classifier_modules_6_parameters_bias_ = None
        return (input_32,)
