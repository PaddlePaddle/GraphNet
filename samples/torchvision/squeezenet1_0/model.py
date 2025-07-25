import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_3_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_squeeze_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_squeeze_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_expand1x1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_expand1x1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_expand3x3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_expand3x3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_3_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_expand3x3_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_squeeze_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_squeeze_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_squeeze_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_squeeze_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_expand1x1_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_expand1x1_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_expand1x1_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_expand1x1_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_expand3x3_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_expand3x3_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_expand3x3_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_expand3x3_parameters_bias_
        )
        l_self_modules_classifier_modules_1_parameters_weight_ = (
            L_self_modules_classifier_modules_1_parameters_weight_
        )
        l_self_modules_classifier_modules_1_parameters_bias_ = (
            L_self_modules_classifier_modules_1_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_parameters_weight_,
            l_self_modules_features_modules_0_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_features_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.nn.functional.max_pool2d(
            input_2, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        input_2 = None
        conv2d_1 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_3_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_3_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_features_modules_3_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_3_modules_squeeze_parameters_bias_ = None
        x = torch.nn.functional.relu(conv2d_1, inplace=True)
        conv2d_1 = None
        conv2d_2 = torch.conv2d(
            x,
            l_self_modules_features_modules_3_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_3_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_3_modules_expand1x1_parameters_bias_
        ) = None
        relu_2 = torch.nn.functional.relu(conv2d_2, inplace=True)
        conv2d_2 = None
        conv2d_3 = torch.conv2d(
            x,
            l_self_modules_features_modules_3_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_3_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x = (
            l_self_modules_features_modules_3_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_3_modules_expand3x3_parameters_bias_ = None
        relu_3 = torch.nn.functional.relu(conv2d_3, inplace=True)
        conv2d_3 = None
        input_4 = torch.cat([relu_2, relu_3], 1)
        relu_2 = relu_3 = None
        conv2d_4 = torch.conv2d(
            input_4,
            l_self_modules_features_modules_4_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_4_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_4 = (
            l_self_modules_features_modules_4_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_squeeze_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(conv2d_4, inplace=True)
        conv2d_4 = None
        conv2d_5 = torch.conv2d(
            x_1,
            l_self_modules_features_modules_4_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_4_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_4_modules_expand1x1_parameters_bias_
        ) = None
        relu_5 = torch.nn.functional.relu(conv2d_5, inplace=True)
        conv2d_5 = None
        conv2d_6 = torch.conv2d(
            x_1,
            l_self_modules_features_modules_4_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_4_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = (
            l_self_modules_features_modules_4_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_expand3x3_parameters_bias_ = None
        relu_6 = torch.nn.functional.relu(conv2d_6, inplace=True)
        conv2d_6 = None
        input_5 = torch.cat([relu_5, relu_6], 1)
        relu_5 = relu_6 = None
        conv2d_7 = torch.conv2d(
            input_5,
            l_self_modules_features_modules_5_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_5_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = (
            l_self_modules_features_modules_5_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_5_modules_squeeze_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(conv2d_7, inplace=True)
        conv2d_7 = None
        conv2d_8 = torch.conv2d(
            x_2,
            l_self_modules_features_modules_5_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_5_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_5_modules_expand1x1_parameters_bias_
        ) = None
        relu_8 = torch.nn.functional.relu(conv2d_8, inplace=True)
        conv2d_8 = None
        conv2d_9 = torch.conv2d(
            x_2,
            l_self_modules_features_modules_5_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_5_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = (
            l_self_modules_features_modules_5_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_5_modules_expand3x3_parameters_bias_ = None
        relu_9 = torch.nn.functional.relu(conv2d_9, inplace=True)
        conv2d_9 = None
        input_6 = torch.cat([relu_8, relu_9], 1)
        relu_8 = relu_9 = None
        input_7 = torch.nn.functional.max_pool2d(
            input_6, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        input_6 = None
        conv2d_10 = torch.conv2d(
            input_7,
            l_self_modules_features_modules_7_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_7_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = (
            l_self_modules_features_modules_7_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_7_modules_squeeze_parameters_bias_ = None
        x_3 = torch.nn.functional.relu(conv2d_10, inplace=True)
        conv2d_10 = None
        conv2d_11 = torch.conv2d(
            x_3,
            l_self_modules_features_modules_7_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_7_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_7_modules_expand1x1_parameters_bias_
        ) = None
        relu_11 = torch.nn.functional.relu(conv2d_11, inplace=True)
        conv2d_11 = None
        conv2d_12 = torch.conv2d(
            x_3,
            l_self_modules_features_modules_7_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_7_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_features_modules_7_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_7_modules_expand3x3_parameters_bias_ = None
        relu_12 = torch.nn.functional.relu(conv2d_12, inplace=True)
        conv2d_12 = None
        input_8 = torch.cat([relu_11, relu_12], 1)
        relu_11 = relu_12 = None
        conv2d_13 = torch.conv2d(
            input_8,
            l_self_modules_features_modules_8_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_8_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = (
            l_self_modules_features_modules_8_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_8_modules_squeeze_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(conv2d_13, inplace=True)
        conv2d_13 = None
        conv2d_14 = torch.conv2d(
            x_4,
            l_self_modules_features_modules_8_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_8_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_8_modules_expand1x1_parameters_bias_
        ) = None
        relu_14 = torch.nn.functional.relu(conv2d_14, inplace=True)
        conv2d_14 = None
        conv2d_15 = torch.conv2d(
            x_4,
            l_self_modules_features_modules_8_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_8_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = (
            l_self_modules_features_modules_8_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_8_modules_expand3x3_parameters_bias_ = None
        relu_15 = torch.nn.functional.relu(conv2d_15, inplace=True)
        conv2d_15 = None
        input_9 = torch.cat([relu_14, relu_15], 1)
        relu_14 = relu_15 = None
        conv2d_16 = torch.conv2d(
            input_9,
            l_self_modules_features_modules_9_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_9_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = (
            l_self_modules_features_modules_9_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_9_modules_squeeze_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(conv2d_16, inplace=True)
        conv2d_16 = None
        conv2d_17 = torch.conv2d(
            x_5,
            l_self_modules_features_modules_9_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_9_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_9_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_9_modules_expand1x1_parameters_bias_
        ) = None
        relu_17 = torch.nn.functional.relu(conv2d_17, inplace=True)
        conv2d_17 = None
        conv2d_18 = torch.conv2d(
            x_5,
            l_self_modules_features_modules_9_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_9_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_features_modules_9_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_9_modules_expand3x3_parameters_bias_ = None
        relu_18 = torch.nn.functional.relu(conv2d_18, inplace=True)
        conv2d_18 = None
        input_10 = torch.cat([relu_17, relu_18], 1)
        relu_17 = relu_18 = None
        conv2d_19 = torch.conv2d(
            input_10,
            l_self_modules_features_modules_10_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_10_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = (
            l_self_modules_features_modules_10_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_10_modules_squeeze_parameters_bias_ = None
        x_6 = torch.nn.functional.relu(conv2d_19, inplace=True)
        conv2d_19 = None
        conv2d_20 = torch.conv2d(
            x_6,
            l_self_modules_features_modules_10_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_10_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_10_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_10_modules_expand1x1_parameters_bias_
        ) = None
        relu_20 = torch.nn.functional.relu(conv2d_20, inplace=True)
        conv2d_20 = None
        conv2d_21 = torch.conv2d(
            x_6,
            l_self_modules_features_modules_10_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_10_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_6 = (
            l_self_modules_features_modules_10_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_10_modules_expand3x3_parameters_bias_ = None
        relu_21 = torch.nn.functional.relu(conv2d_21, inplace=True)
        conv2d_21 = None
        input_11 = torch.cat([relu_20, relu_21], 1)
        relu_20 = relu_21 = None
        input_12 = torch.nn.functional.max_pool2d(
            input_11, 3, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        input_11 = None
        conv2d_22 = torch.conv2d(
            input_12,
            l_self_modules_features_modules_12_modules_squeeze_parameters_weight_,
            l_self_modules_features_modules_12_modules_squeeze_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_12 = (
            l_self_modules_features_modules_12_modules_squeeze_parameters_weight_
        ) = l_self_modules_features_modules_12_modules_squeeze_parameters_bias_ = None
        x_7 = torch.nn.functional.relu(conv2d_22, inplace=True)
        conv2d_22 = None
        conv2d_23 = torch.conv2d(
            x_7,
            l_self_modules_features_modules_12_modules_expand1x1_parameters_weight_,
            l_self_modules_features_modules_12_modules_expand1x1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_12_modules_expand1x1_parameters_weight_ = (
            l_self_modules_features_modules_12_modules_expand1x1_parameters_bias_
        ) = None
        relu_23 = torch.nn.functional.relu(conv2d_23, inplace=True)
        conv2d_23 = None
        conv2d_24 = torch.conv2d(
            x_7,
            l_self_modules_features_modules_12_modules_expand3x3_parameters_weight_,
            l_self_modules_features_modules_12_modules_expand3x3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_7 = (
            l_self_modules_features_modules_12_modules_expand3x3_parameters_weight_
        ) = l_self_modules_features_modules_12_modules_expand3x3_parameters_bias_ = None
        relu_24 = torch.nn.functional.relu(conv2d_24, inplace=True)
        conv2d_24 = None
        input_13 = torch.cat([relu_23, relu_24], 1)
        relu_23 = relu_24 = None
        input_14 = torch.nn.functional.dropout(input_13, 0.5, False, False)
        input_13 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        input_17 = torch.nn.functional.adaptive_avg_pool2d(input_16, (1, 1))
        input_16 = None
        flatten = torch.flatten(input_17, 1)
        input_17 = None
        return (flatten,)
