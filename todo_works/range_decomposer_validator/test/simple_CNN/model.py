import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_conv1_parameters_weight_ = (
            L_self_modules_conv1_parameters_weight_
        )
        l_self_modules_conv1_parameters_bias_ = L_self_modules_conv1_parameters_bias_
        l_self_modules_conv2_parameters_weight_ = (
            L_self_modules_conv2_parameters_weight_
        )
        l_self_modules_conv2_parameters_bias_ = L_self_modules_conv2_parameters_bias_
        l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
        l_self_modules_fc1_parameters_bias_ = L_self_modules_fc1_parameters_bias_
        l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
        l_self_modules_fc2_parameters_bias_ = L_self_modules_fc2_parameters_bias_

        # --- Subgraph 0 ---
        # conv1 -> relu -> pool1
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_conv1_parameters_weight_,
            l_self_modules_conv1_parameters_bias_,
            (1, 1),  # stride
            (1, 1),  # padding
            (1, 1),  # dilation
            1,  # groups
        )
        l_x_ = (
            l_self_modules_conv1_parameters_weight_
        ) = l_self_modules_conv1_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.nn.functional.max_pool2d(input_2, 2, 2, 0, 1, ceil_mode=False)
        input_2 = None

        # --- Subgraph 1 ---
        # conv2 -> relu -> pool2
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_conv2_parameters_weight_,
            l_self_modules_conv2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_conv2_parameters_weight_
        ) = l_self_modules_conv2_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch.nn.functional.max_pool2d(input_5, 2, 2, 0, 1, ceil_mode=False)
        input_5 = None

        # --- Subgraph 2 ---
        # flatten -> fc1 -> relu -> fc2
        input_7 = torch.flatten(input_6, 1)
        input_6 = None
        input_8 = torch._C._nn.linear(
            input_7,
            l_self_modules_fc1_parameters_weight_,
            l_self_modules_fc1_parameters_bias_,
        )
        input_7 = (
            l_self_modules_fc1_parameters_weight_
        ) = l_self_modules_fc1_parameters_bias_ = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        input_10 = torch._C._nn.linear(
            input_9,
            l_self_modules_fc2_parameters_weight_,
            l_self_modules_fc2_parameters_bias_,
        )
        input_9 = (
            l_self_modules_fc2_parameters_weight_
        ) = l_self_modules_fc2_parameters_bias_ = None

        return (input_10,)
