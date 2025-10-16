import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_conv1_parameters_weight_ = (
            L_self_modules_conv1_parameters_weight_
        )
        l_self_modules_conv1_parameters_bias_ = L_self_modules_conv1_parameters_bias_

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

        return (input_3,)
