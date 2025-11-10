import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        l_l_self_modules_fc_parameters_bias_,
        l_l_self_modules_fc_parameters_weight_,
        l_l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
        l_l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
        l_l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
        l_l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
        l_l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
        l_l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
        l_l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_,
        l_l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_,
        l_l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
        l_l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_,
        l_l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
        l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
        l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
        l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
        l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
        l_l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
        l_l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
        l_l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
        l_l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
        l_l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
        l_l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
        l_l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
        l_l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
        l_l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
        l_l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
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
        x_10,
        x_15,
    ):
        x_15 += x_10
        x_16 = x_15
        x_15 = x_10 = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_20 = l_l_self_modules_layer2_modules_0_modules_conv2_parameters_weight_ = None
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_l_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_l_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_l_self_modules_layer2_modules_0_modules_bn2_parameters_weight_
        ) = l_l_self_modules_layer2_modules_0_modules_bn2_parameters_bias_ = None
        input_1 = torch.conv2d(
            x_17,
            l_l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_22 += input_2
        x_23 = x_22
        x_22 = input_2 = None
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_27 = l_l_self_modules_layer2_modules_1_modules_conv2_parameters_weight_ = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_l_self_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_l_self_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_l_self_modules_layer2_modules_1_modules_bn2_parameters_weight_
        ) = l_l_self_modules_layer2_modules_1_modules_bn2_parameters_bias_ = None
        return (
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
        )
