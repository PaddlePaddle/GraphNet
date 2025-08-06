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
        L_self_modules_pre_logits_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pre_logits_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_pre_logits_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_pre_logits_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_pre_logits_modules_fc1_parameters_weight_ = (
            L_self_modules_pre_logits_modules_fc1_parameters_weight_
        )
        l_self_modules_pre_logits_modules_fc1_parameters_bias_ = (
            L_self_modules_pre_logits_modules_fc1_parameters_bias_
        )
        l_self_modules_pre_logits_modules_fc2_parameters_weight_ = (
            L_self_modules_pre_logits_modules_fc2_parameters_weight_
        )
        l_self_modules_pre_logits_modules_fc2_parameters_bias_ = (
            L_self_modules_pre_logits_modules_fc2_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
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
        x = torch.conv2d(
            input_25,
            l_self_modules_pre_logits_modules_fc1_parameters_weight_,
            l_self_modules_pre_logits_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = (
            l_self_modules_pre_logits_modules_fc1_parameters_weight_
        ) = l_self_modules_pre_logits_modules_fc1_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(x, inplace=True)
        x = None
        x_2 = torch.nn.functional.dropout(x_1, 0.0, False, False)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_pre_logits_modules_fc2_parameters_weight_,
            l_self_modules_pre_logits_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = (
            l_self_modules_pre_logits_modules_fc2_parameters_weight_
        ) = l_self_modules_pre_logits_modules_fc2_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        x_5 = torch.nn.functional.adaptive_avg_pool2d(x_4, 1)
        x_4 = None
        x_6 = x_5.flatten(1, -1)
        x_5 = None
        x_7 = torch.nn.functional.dropout(x_6, 0.0, False, False)
        x_6 = None
        x_8 = torch._C._nn.linear(
            x_7,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_7 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_8,)
