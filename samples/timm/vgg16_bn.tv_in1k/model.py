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
        x = torch.conv2d(
            input_44,
            l_self_modules_pre_logits_modules_fc1_parameters_weight_,
            l_self_modules_pre_logits_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_44 = (
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
