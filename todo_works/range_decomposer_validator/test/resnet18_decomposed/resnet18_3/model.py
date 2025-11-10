import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        l_l_self_modules_fc_parameters_bias_,
        l_l_self_modules_fc_parameters_weight_,
        l_l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
        l_l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
        l_l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
        l_l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
        l_l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
        l_l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
        l_l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
        l_l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
        l_l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
        l_l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
        l_l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
        l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
        l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
        l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
        l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
        l_l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
        l_l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
        l_l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
        l_l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
        l_l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
        l_l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
        l_l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
        l_l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
        l_l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
        l_l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
        l_l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
        l_l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
        l_l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
        l_l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
        l_l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
        l_l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
        l_l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
        l_l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
        l_l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
        l_l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
        l_l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
        l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
        l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
        l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
        l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
        l_l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
        l_l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
        l_l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
        l_l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
        l_l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
        l_l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
        l_l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
        l_l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
        l_l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
        l_l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
        x_24,
        x_29,
    ):
        x_29 += x_24
        x_30 = x_29
        x_29 = x_24 = None
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = (
            l_l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_34 = l_l_self_modules_layer3_modules_0_modules_conv2_parameters_weight_ = None
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_l_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_l_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_l_self_modules_layer3_modules_0_modules_bn2_parameters_weight_
        ) = l_l_self_modules_layer3_modules_0_modules_bn2_parameters_bias_ = None
        input_3 = torch.conv2d(
            x_31,
            l_l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_36 += input_4
        x_37 = x_36
        x_36 = input_4 = None
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = (
            l_l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_41 = l_l_self_modules_layer3_modules_1_modules_conv2_parameters_weight_ = None
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_l_self_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_l_self_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_l_self_modules_layer3_modules_1_modules_bn2_parameters_weight_
        ) = l_l_self_modules_layer3_modules_1_modules_bn2_parameters_bias_ = None
        x_43 += x_38
        x_44 = x_43
        x_43 = x_38 = None
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = (
            l_l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_48 = l_l_self_modules_layer4_modules_0_modules_conv2_parameters_weight_ = None
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = (
            l_l_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_l_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_l_self_modules_layer4_modules_0_modules_bn2_parameters_weight_
        ) = l_l_self_modules_layer4_modules_0_modules_bn2_parameters_bias_ = None
        input_5 = torch.conv2d(
            x_45,
            l_l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_50 += input_6
        x_51 = x_50
        x_50 = input_6 = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_l_self_modules_layer4_modules_1_modules_conv2_parameters_weight_ = None
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = (
            l_l_self_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_l_self_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_l_self_modules_layer4_modules_1_modules_bn2_parameters_weight_
        ) = l_l_self_modules_layer4_modules_1_modules_bn2_parameters_bias_ = None
        x_57 += x_52
        x_58 = x_57
        x_57 = x_52 = None
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.nn.functional.adaptive_avg_pool2d(x_59, 1)
        x_59 = None
        x_61 = x_60.flatten(1, -1)
        x_60 = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_l_self_modules_fc_parameters_weight_,
            l_l_self_modules_fc_parameters_bias_,
        )
        x_61 = (
            l_l_self_modules_fc_parameters_weight_
        ) = l_l_self_modules_fc_parameters_bias_ = None
        return (x_62,)
