import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_6_parameters_weight_ = (
            L_self_modules_features_modules_6_parameters_weight_
        )
        l_self_modules_features_modules_6_parameters_bias_ = (
            L_self_modules_features_modules_6_parameters_bias_
        )
        l_self_modules_features_modules_8_parameters_weight_ = (
            L_self_modules_features_modules_8_parameters_weight_
        )
        l_self_modules_features_modules_8_parameters_bias_ = (
            L_self_modules_features_modules_8_parameters_bias_
        )
        l_self_modules_features_modules_11_parameters_weight_ = (
            L_self_modules_features_modules_11_parameters_weight_
        )
        l_self_modules_features_modules_11_parameters_bias_ = (
            L_self_modules_features_modules_11_parameters_bias_
        )
        l_self_modules_features_modules_13_parameters_weight_ = (
            L_self_modules_features_modules_13_parameters_weight_
        )
        l_self_modules_features_modules_13_parameters_bias_ = (
            L_self_modules_features_modules_13_parameters_bias_
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
        input_3 = torch.nn.functional.max_pool2d(
            input_2, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
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
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch.nn.functional.max_pool2d(
            input_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_features_modules_6_parameters_weight_,
            l_self_modules_features_modules_6_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = (
            l_self_modules_features_modules_6_parameters_weight_
        ) = l_self_modules_features_modules_6_parameters_bias_ = None
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
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
        input_10 = torch.nn.functional.relu(input_9, inplace=True)
        input_9 = None
        input_11 = torch.nn.functional.max_pool2d(
            input_10, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
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
        input_13 = torch.nn.functional.relu(input_12, inplace=True)
        input_12 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_features_modules_13_parameters_weight_,
            l_self_modules_features_modules_13_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_13 = (
            l_self_modules_features_modules_13_parameters_weight_
        ) = l_self_modules_features_modules_13_parameters_bias_ = None
        input_15 = torch.nn.functional.relu(input_14, inplace=True)
        input_14 = None
        input_16 = torch.nn.functional.max_pool2d(
            input_15, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_15 = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_features_modules_16_parameters_weight_,
            l_self_modules_features_modules_16_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_16 = (
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
        input_20 = torch.nn.functional.relu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.nn.functional.max_pool2d(
            input_20, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_20 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_21, (7, 7))
        input_21 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_22 = torch._C._nn.linear(
            x_1,
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
        )
        x_1 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.nn.functional.dropout(input_23, 0.5, False, False)
        input_23 = None
        input_25 = torch._C._nn.linear(
            input_24,
            l_self_modules_classifier_modules_3_parameters_weight_,
            l_self_modules_classifier_modules_3_parameters_bias_,
        )
        input_24 = (
            l_self_modules_classifier_modules_3_parameters_weight_
        ) = l_self_modules_classifier_modules_3_parameters_bias_ = None
        input_26 = torch.nn.functional.relu(input_25, inplace=True)
        input_25 = None
        input_27 = torch.nn.functional.dropout(input_26, 0.5, False, False)
        input_26 = None
        input_28 = torch._C._nn.linear(
            input_27,
            l_self_modules_classifier_modules_6_parameters_weight_,
            l_self_modules_classifier_modules_6_parameters_bias_,
        )
        input_27 = (
            l_self_modules_classifier_modules_6_parameters_weight_
        ) = l_self_modules_classifier_modules_6_parameters_bias_ = None
        return (input_28,)
