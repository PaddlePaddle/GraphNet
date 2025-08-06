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
        x = torch.conv2d(
            input_21,
            l_self_modules_pre_logits_modules_fc1_parameters_weight_,
            l_self_modules_pre_logits_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = (
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
